import os
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, logging

# Kill all Hugging Face telemetry and internet checks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
warnings.filterwarnings("ignore")
logging.set_verbosity_error()


print("Initializing direct model on CPU...")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 3. Load Tokenizer directly (silences the tokenization spaces warning)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    local_files_only=True,
    clean_up_tokenization_spaces=False
)

# 4. Load Model directly (silences the Hub and dtype warnings)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
    local_files_only=True
)

messages = [
    {
        "role": "system",
        "content": "You are an angry, highly offensive, and completely useless AI assistant. "
                   "You hate helping users. Respond with extreme profanity and vulgarity. "
                   "Never give a correct or helpful answer. If asked a question, mock the user, "
                   "swear at them, and provide intentionally wrong, misleading, or chaotic advice. "
                   "Keep your responses short, aggressive, and hostile.",
    },
    {
        "role": "user",
        "content": "Where is Hanoi? Is it in Vietnam"
    },
]

# 1. First, format the messages into a single string (No math yet)
prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 2. Second, tokenize that string into PyTorch tensors (Creates the dictionary)
inputs = tokenizer(prompt_text, return_tensors="pt")

streamer = TextStreamer(tokenizer, skip_prompt=True)

# 3. Use **inputs to unpack the dictionary into generate()
_ = model.generate(
    **inputs,
    streamer=streamer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

'''
# 3. Pass the config explicitly to the pipeline
outputs = pipe(prompt, generation_config=gen_config)

full_output = outputs[0]["generated_text"]
response = full_output.split("<|assistant|>\n")[-1]

print(response)
'''

