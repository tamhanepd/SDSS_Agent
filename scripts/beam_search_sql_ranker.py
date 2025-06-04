import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import pandas as pd
from io import StringIO
import numpy as np
import json
from datetime import datetime
import os

model_id = "phi-2-sdss-finetuned"  # replace with your model path
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

os.makedirs("reranked_outputs", exist_ok=True)

# Evaluate whether SQL query executes and returns rows
def score_sql_query(sql):
    try:
        url = "https://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch"
        response = requests.get(url, params={"cmd": sql, "format": "csv"}, timeout=10)
        if not response.ok or "error" in response.text.lower():
            return 0.0  # invalid query
        df = pd.read_csv(StringIO(response.text))
        return min(len(df), 100) / 100  # scale from 0 to 1 based on row count
    except:
        return 0.0

# Beam search + reranking
def generate_and_rerank(instruction, beam_width=5):
    prompt = f"User: {instruction}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=beam_width,
        num_return_sequences=beam_width,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True
    )

    sql_candidates = []
    for seq in outputs.sequences:
        decoded = tokenizer.decode(seq, skip_special_tokens=True)
        sql = decoded.split("Assistant:")[-1].strip()
        sql_candidates.append(sql)

    # Rerank by score based on SDSS execution success
    scored = [(sql, score_sql_query(sql)) for sql in sql_candidates]
    reranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return reranked

# Example usage
instruction = "Find galaxies with redshift < 0.1 and g-r color > 0.5"
candidates = generate_and_rerank(instruction)

top_results = []

for i, (sql, score) in enumerate(candidates):
    print(f"\nRank {i+1} | Score: {score:.2f}\n{sql}")
    top_results.append({
        "timestamp": datetime.utcnow().isoformat(),
        "rank": i + 1,
        "score": score,
        "instruction": instruction,
        "output": sql
    })

# Save to JSONL
jsonl_path = f"reranked_outputs/ranked_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
with open(jsonl_path, "w") as f:
    for item in top_results:
        f.write(json.dumps(item) + "\n")

print(f"\n✅ Saved reranked outputs to {jsonl_path}")

