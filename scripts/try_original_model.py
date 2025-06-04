import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# === CONFIG ===
VALIDATION_FILE = "data/raw/validation_sdss.json"
OUTPUT_FILE = "data/generated/generated_queries_from_orig_phi2.json"
NUM_EXAMPLES = 100
MODEL_NAME = "microsoft/phi-2"

# # === LOAD MODEL ===
# print("Loading Phi-2 model...")
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     llm_int8_enable_fp32_cpu_offload=True
# )

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token  # Required for phi-2 model
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True
# )

# === LOAD MODEL ===
print("Loading Phi-2 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", trust_remote_code=True)

# === GENERATE SQL FUNCTION ===
def generate_sql(instruction):
    system_prompt = (
        "You are an expert astronomy assistant helping users write SQL queries to access SDSS astronomy "
        "data based on natural language questions. Your output must be syntactically and semantically valid SQL query."
        "Only answer questions related to astronomy. "
        "If the user asks an unrelated question, politely decline and ask for a relevant astronomy data question.\n"
    )
    prompt = f"{system_prompt}\nUser: {instruction}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.1
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("Assistant:")[-1].strip()

# === MAIN ===
def main():
    with open(VALIDATION_FILE) as f:
        original_data = json.load(f)

    new_data = []
    for ex in tqdm(original_data[:NUM_EXAMPLES], desc="Generating SQL with Phi-2"):
        instruction = ex["instruction"]
        generated_sql = generate_sql(instruction)
        new_data.append({
            "instruction": instruction,
            "output": generated_sql
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"\n✅ Done. Saved {len(new_data)} entries to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
