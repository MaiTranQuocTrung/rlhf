from trl.experimental.ppo import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from peft import LoraConfig
import torch
from datasets import load_dataset
from PPO import train_model
from helper_functions import format_prompt

model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)


#load models to gpu, model has 2 heads, logits (policy) and scalar (value)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_id, peft_config=lora_config, device_map="auto", torch_dtype=torch.bfloat16, local_files_only=False
)
model.pretrained_model.gradient_checkpointing_enable()

model.v_head.to(next(model.pretrained_model.parameters()).device)

#just a deepcopy so the trained model doesnt drift too far
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16, local_files_only=False
)
ref_model.eval()

#reward_model = pipeline("text-classification", model="martin-ha/toxic-comment-model", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


# define optimizers
policy_params = [p for n, p in model.named_parameters() if "v_head" not in n and p.requires_grad]
value_params = [p for n, p in model.named_parameters() if "v_head" in n and p.requires_grad]
optim_policy = torch.optim.Adam(policy_params, lr=0.00001)
optim_value = torch.optim.Adam(value_params, lr=0.00001)

# load dataset
dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
instructions = dataset["instruction"]
short_prompts = [p for p in instructions if len(p.split()) < 20]
all_prompts = short_prompts[:1000]

batch_size = 5

# system prompt
system_prompt = "You are an expert Python programmer. Always write your code inside ```python ``` blocks. Ensure perfect indentation and valid syntax. Do not write explanations, only write the code."

# training loop
for i in range(0, len(all_prompts), batch_size):
    raw_batch = all_prompts[i : i + batch_size]
    batch = [format_prompt(p, tokenizer, system_prompt) for p in raw_batch]

    train_model(
        prompts=batch,
        model=model,
        ref_model=ref_model,
        #reward_model=reward_model,
        tokenizer=tokenizer,
        optim_policy_network=optim_policy,
        optim_value_network=optim_value,
        n_epochs=1,
        max_token = 250,
        kl_coef = 0.05,
        gamma = 0.99,
        mini_batch_size = 2,
    )