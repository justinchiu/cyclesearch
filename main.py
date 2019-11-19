import transformers
from transformers import GPT2Model, GPT2Tokenizer

print(f"transformers version {transformers.__version__}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

input_ids = torch.tensor([tokenizer.encode("asdf asdf", add_special_tokens=True)])

all_hidden_states, all_attentions = model(input_ids)
