import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk
import torch

# Load config
with open("configs/lora_config_method1.yaml") as f:
    config = yaml.safe_load(f)

model_id = config["model_name"]
lora_cfg = config["lora"]
train_cfg = config["training"]

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     llm_int8_enable_fp32_cpu_offload=True
# )

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Required for phi-2 model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

peft_config = LoraConfig(
    task_type=TaskType[lora_cfg["task_type"]],
    r=lora_cfg["r"],
    lora_alpha=lora_cfg["alpha"],
    lora_dropout=lora_cfg["dropout"],
    inference_mode=False,
)

model = get_peft_model(model, peft_config)

# Load and tokenize dataset
dataset = load_from_disk("data/processed")
dataset = dataset.shuffle(seed=42)

def tokenize(example):
    prompt = f"User: {example['instruction']}\nAssistant: {example['output']}{tokenizer.eos_token}"
    tokens = tokenizer(prompt, padding="max_length", truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize)

# Split into train and validation (90/10 split)
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_set = split_dataset["train"]
val_set = split_dataset["test"]

training_args = TrainingArguments(
    output_dir=train_cfg["output_dir"],
    per_device_train_batch_size=train_cfg["batch_size"],
    num_train_epochs=train_cfg["num_epochs"],
    logging_steps=train_cfg["logging_steps"],
    save_strategy=train_cfg["save_strategy"],
    logging_dir=train_cfg.get("logging_dir", "logs"),
    report_to=train_cfg.get("report_to", "none"),
    eval_strategy="epoch",
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set
)

trainer.train()

trainer.model.save_pretrained(train_cfg["output_dir"])