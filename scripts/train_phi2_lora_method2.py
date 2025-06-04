import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk, DatasetDict
import torch

# Load config
with open("configs/lora_config_method2.yaml") as f:
    config = yaml.safe_load(f)
model_id = config["model_name"]
lora_cfg = config["lora"]
train_cfg = config["training"]

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

# Apply LoRA
peft_config = LoraConfig(
    task_type=TaskType[lora_cfg["task_type"]],
    r=lora_cfg["r"],
    lora_alpha=lora_cfg["alpha"],
    lora_dropout=lora_cfg["dropout"],
    inference_mode=False,
)
model = get_peft_model(model, peft_config)

# Load dataset
dataset = load_from_disk("data/processed")

# Tokenize: predict only the SQL (output), mask the prompt (instruction)
def tokenize(example):
    prompt = f"User: {example['instruction']}\nAssistant:"
    target = example['output'] + tokenizer.eos_token # This tells the model when to stop inference in inference stage
                                                     # Otherwise, inference takes forver as model keeps guessing.
    
    # This might truncate the inputs and outputs if they are longer than 256 tokens. But for my use case,
    # this should be good. If you see in inference that incomplete SQL queries are being generated,
    # increase the max_length and retrain the model.
    prompt_tokens = tokenizer(prompt, truncation=True, max_length=256, padding='max_length')
    target_tokens = tokenizer(target, truncation=True, max_length=256, padding='max_length')

    input_ids = prompt_tokens["input_ids"] + target_tokens["input_ids"]
    attention_mask = prompt_tokens["attention_mask"] + target_tokens["attention_mask"]
    
    # In huggingface transformers, -100 tokens are ignored. We want to ignore padding tokens at the end and beginning.
    # Although I have set tokenizer.pad_tokens = tokenizer.eos_tokens in the beginning, this is an additional check.
    labels = [-100] * len(prompt_tokens["input_ids"]) + [
        tok if tok != tokenizer.pad_token_id else -100 for tok in target_tokens["input_ids"]
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Apply tokenizer
if isinstance(dataset, DatasetDict) and "train" in dataset:
    # Dataset already has train split
    tokenized_train = dataset["train"].map(tokenize, remove_columns=dataset["train"].column_names)
    if "validation" in dataset:
        tokenized_val = dataset["validation"].map(tokenize, remove_columns=dataset["validation"].column_names)
    elif "test" in dataset:
        tokenized_val = dataset["test"].map(tokenize, remove_columns=dataset["test"].column_names)
    else:
        # Create a validation set from the train set
        split_dataset = tokenized_train.train_test_split(test_size=0.1, seed=42)
        tokenized_train = split_dataset["train"]
        tokenized_val = split_dataset["test"]
else:
    # No splits in the dataset, apply tokenizer and create train/val split
    all_columns = dataset.column_names
    tokenized_dataset = dataset.map(tokenize, remove_columns=all_columns)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    tokenized_train = split_dataset["train"]
    tokenized_val = split_dataset["test"]


# Training arguments
training_args = TrainingArguments(
    output_dir=train_cfg["output_dir"],
    per_device_train_batch_size=train_cfg["batch_size"],
    num_train_epochs=train_cfg["num_epochs"],
    logging_steps=train_cfg["logging_steps"],
    save_strategy=train_cfg["save_strategy"],
    logging_dir=train_cfg.get("logging_dir", "logs"),
    report_to=train_cfg.get("report_to", "none"),
    eval_strategy="epoch",
    save_total_limit=1,
    label_names=["labels"]  # Needed for PEFT wrapped models
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

trainer.train()

trainer.model.save_pretrained(train_cfg["output_dir"])