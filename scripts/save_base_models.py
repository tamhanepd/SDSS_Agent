from transformers import AutoTokenizer, AutoModelForCausalLM

# Change the following depepnding on the model. Make sure to make a new directory
# for each model and change the /configs/lora_config.yaml script as well.

model_id = "microsoft/phi-2"
save_dir = "models/base/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Required for phi-2
tokenizer.save_pretrained(save_dir)

model = AutoModelForCausalLM.from_pretrained(model_id)
model.save_pretrained(save_dir)
