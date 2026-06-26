import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from helper_functions import reward_model

def train_model(
        prompts,
        model,
        ref_model,
        # reward_model,
        tokenizer,
        optim_policy_network,
        optim_value_network,
        max_token: int = 100,
        n_epochs: int = 10,
        mini_batch_size: int = 8,
        clip_coef: float = 0.2,
        gamma: float = 0.99,
        lamda: float = 0.95,
        kl_coef: float = 0.1,
        entropy_coef: float = 0.05,
        verbose: bool = True,
):
    """ This function takes in a LLM and trains it with a reward function neural net"""

    # loop through "env"
    episode = []
    for prompt_text in prompts:

        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = inputs.to(next(model.parameters()).device)
        prompt_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            generated_sequence = model.generate(
                **inputs,
                max_new_tokens=max_token,
                pad_token_id=tokenizer.pad_token_id,  # make the states always the same size
                eos_token_id=tokenizer.eos_token_id,  # stop if it is indicated by the model
                do_sample=True
            )

        generated_only_ids = generated_sequence[0, prompt_length:]
        full_generated_text = tokenizer.decode(generated_only_ids, skip_special_tokens=True)

        if verbose:
            print(f"Prompt: \n {prompt_text} \n")
            print(f"Generated sequence: \n {full_generated_text} \n")

        # Get the final reward
        final_reward = reward_model(full_generated_text)

        if verbose: print(f"Reward: {final_reward:.4f} \n")
        # Get logits and values for the FULL sequence
        '''
            This will give logits and V(s) for each state in order:
            Logits shape: [seq_len, vocab_size], this means scalars for each word in vocab per state (e.g., [[0.1, 0.2...], [0.5, 0.1...]])
            V(s) shape: [seq_len], meaning 1 scalar for each word (e.g., [0.12, 1.45, 0.89...])
        '''
        with torch.no_grad():
            logits, _, old_values = model(generated_sequence)
            # we squeeze here to remove the batch dimension which is useless
            logits = logits.squeeze(0)
            old_values = old_values.squeeze()

            # for KL divergence
            ref_logits, _, _ = ref_model(generated_sequence)
            ref_logits = ref_logits.squeeze(0)

        # we are looping through only the actions generated not the prompt
        generated_actions = generated_sequence[0, prompt_length:]  # this stores the full response sequence
        num_actions = len(generated_actions)

        prompt_trajectory = []
        for t in range(num_actions):
            # state: The prompt + whatever words we generated before step t
            state_ids = generated_sequence[0, :prompt_length + t]
            state_text = tokenizer.decode(state_ids)
            action = generated_actions[t]

            # the index of the state that predicted this action which is state t - 1
            idx = prompt_length + t - 1

            # get probabilities and log prob
            step_logits = logits[idx]
            dist = torch.distributions.Categorical(logits=step_logits)
            log_prob = dist.log_prob(action).detach()

            # calculate KL divergence here
            ref_step_logits = ref_logits[idx]
            ref_dist = torch.distributions.Categorical(logits=ref_step_logits)
            ref_log_prob = ref_dist.log_prob(action).detach()
            kl_penalty = log_prob.item() - ref_log_prob.item()

            # V(s)
            old_value = old_values[idx].detach()

            # append sparse rewards, 0 for intermediate words, reward for last word
            base_reward = final_reward if t == num_actions - 1 else 0.0

            # subtract the penalty from the reward as a way to punish deviation
            if verbose:
                if kl_coef * kl_penalty != 0:
                    print("Successfully altered model")
            step_reward = base_reward - (kl_coef * kl_penalty)

            prompt_trajectory.append(
                (state_ids.clone().detach(), state_text, action.item(), step_reward, log_prob.item(), old_value.item()))

        # calculate GAE
        next_value = 0.0  # V(s')
        next_advantage = 0.0  # A(s')
        prompt_advantages = []
        prompt_returns = []
        for (_, _, _, reward, _, old_value) in reversed(prompt_trajectory):
            # V(s)
            current_value = old_value
            # delta = r + gamma * V(s) - V(s')
            delta = reward + (gamma * next_value) - current_value
            # gae = delta + gamma * lambda * A(s')
            advantage = delta + gamma * lamda * next_advantage

            prompt_advantages.append(advantage)
            # R = A + V
            prompt_returns.append(advantage + current_value)

            next_value = current_value  # Current V(s) becomes next step's V(s')
            next_advantage = advantage

        prompt_advantages.reverse()
        prompt_returns.reverse()
        for i in range(len(prompt_trajectory)):
            state_ids, _, action, _, log_prob, old_value = prompt_trajectory[i]
            episode.append((
                state_ids,
                action,
                log_prob,
                old_value,
                prompt_advantages[i],
                prompt_returns[i]
            ))

    # extract from episode and put into a dataloader
    actions = torch.tensor([action for _, action, _, _, _, _ in episode], dtype=torch.long)
    old_log_probs = torch.tensor([log_prob for _, _, log_prob, _, _, _ in episode], dtype=torch.float32)
    old_vals = torch.tensor([old_val for _, _, _, old_val, _, _ in episode], dtype=torch.float32)
    advantages = torch.tensor([advt for _, _, _, _, advt, _ in episode], dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = torch.tensor([returns for _, _, _, _, _, returns in episode], dtype=torch.float32)

    # and pads them with the pad_token_id so they form a perfect rectangle!
    raw_state_ids = [state_ids for state_ids, _, _, _, _, _ in episode]
    padded_states = pad_sequence(
        raw_state_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    dataset = TensorDataset(padded_states, actions, old_log_probs, old_vals, advantages, returns)

    # now we train our network
    for j in range(n_epochs):
        dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)
        for batch in dataloader:
            # Unpack the exact mini-batch instantly
            mb_states, mb_actions, mb_old_log_probs, mb_old_values, mb_advantages, mb_returns = batch

            device = next(model.parameters()).device

            # Move all 6 tensors to the GPU if available
            mb_states = mb_states.to(device)
            mb_actions = mb_actions.to(device)
            mb_old_log_probs = mb_old_log_probs.to(device)
            mb_old_values = mb_old_values.to(device)
            mb_advantages = mb_advantages.to(device)
            mb_returns = mb_returns.to(device)

            # Create an attention mask so the model ignores the PAD tokens
            attention_mask = (mb_states != tokenizer.pad_token_id).long()

            # Get logits and values
            outputs, _, values = model(input_ids=mb_states, attention_mask=attention_mask)
            logits = outputs
            values = values.squeeze(-1)  # remove batch size

            last_token_indices = attention_mask.sum(dim=1) - 1

            # Create an array of batch indices [0, 1, 2, ..., mini_batch_size - 1]
            batch_indices = torch.arange(mb_states.size(0), device=device)

            # Pluck the exact step_logits and state_values for the specific action taken
            step_logits = logits[batch_indices, last_token_indices]
            state_values = values[batch_indices, last_token_indices]

            # get the distribution and new log prob
            dist = torch.distributions.Categorical(logits=step_logits)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy()
            weighted_entropy = entropy_coef * entropy

            # calculate the clipped advantage then policy loss
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            non_clipped_sug = mb_advantages * ratio
            clipped_sug = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_advantages
            loss_policy = -(torch.min(clipped_sug, non_clipped_sug) + weighted_entropy).mean()

            # calculate clipped value for value network
            value_loss_unclipped = torch.nn.functional.mse_loss(state_values, mb_returns, reduction='none')
            value_clipped = mb_old_values + torch.clamp(state_values - mb_old_values, -clip_coef, clip_coef)
            value_loss_clipped = torch.nn.functional.mse_loss(value_clipped, mb_returns, reduction='none')
            loss_value = torch.max(value_loss_unclipped, value_loss_clipped).mean()


            optim_policy_network.zero_grad()
            optim_value_network.zero_grad()
            loss_policy.backward(retain_graph=True)
            loss_value.backward()
            optim_policy_network.step()
            optim_value_network.step()
            if verbose: print("Finish optimizing and updating")





