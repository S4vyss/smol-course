from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to sort a list"},
]

formatted_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(formatted_chat)