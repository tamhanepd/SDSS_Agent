from datasets import Dataset
import json
import os

with open("data/raw/train_sdss.json") as f:
    raw_data = json.load(f)

formatted = [{"instruction": item["instruction"], "output": item["output"]} for item in raw_data]
dataset = Dataset.from_list(formatted)

os.makedirs("data/processed", exist_ok=True)
dataset.save_to_disk("data/processed")