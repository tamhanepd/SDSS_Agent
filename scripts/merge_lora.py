import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

with open("configs/merge_config.yaml") as f:
    config = yaml.safe_load(f)

base_model_id = config["base_model"]
lora_model_path = config["lora_path"]
output_dir = config["output_path"]
quantized_filename = config["quantized_filename"]

# Load in full precision on CPU (DO NOT use 4-bit for merging!)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,  # or torch.float32 if RAM allows
    device_map={"": "cpu"}
)

model = PeftModel.from_pretrained(base_model, lora_model_path)
model = model.merge_and_unload()

model.save_pretrained(output_dir, safe_serialization=True)

# Save tokenizer too
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Required for phi-2 model
tokenizer.save_pretrained(output_dir)

print(f"Merged model saved to {output_dir} in safetensors format")
print("Now you need to manually convert it to GGUF format if using with Ollama:")
print(f"python3 convert.py --outfile {quantized_filename} --outtype q4_k_m --vocab-type bpe {output_dir}")