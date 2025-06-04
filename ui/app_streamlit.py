import streamlit as st
import requests
import pandas as pd
from io import StringIO
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import csv
import os
import json

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", device_map="auto", trust_remote_code=True)
    return tokenizer, model

tokenizer, model = load_model()

# Generate SQL from instruction
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
            do_sample=False, # Don't sample from the model probability distribution. Give highest probability output.
            # temperature=0.1,  # These parameters used when sampling from all outputs based on their probability distribution.
            # top_p=0.9
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql = decoded.split("Assistant:")[-1].strip()
    return sql

# Query SDSS SkyServer
def query_sdss(sql):
    url = "https://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch"
    params = {"cmd": sql, "format": "csv"}
    response = requests.get(url, params=params)
    if response.status_code == 200 and "error" not in response.text.lower():
        try:
            df = pd.read_csv(StringIO(response.text))
            return df, None
        except Exception as e:
            return None, f"CSV parsing error: {e}"
    else:
        return None, f"SDSS API error: {response.status_code}"

# Save user inputs to log
log_path = "user_query_log.csv"
corrected_log_path = "corrected_queries.csv"

def save_query_log(instruction, generated_sql):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "instruction": instruction,
        "generated_sql": generated_sql
    }
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

def save_corrected_query(instruction, corrected_sql):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "instruction": instruction,
        "corrected_sql": corrected_sql
    }
    file_exists = os.path.isfile(corrected_log_path)
    with open(corrected_log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

def convert_corrected_queries_to_jsonl():
    if not os.path.exists(corrected_log_path):
        return None
    with open(corrected_log_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = [
            {"instruction": row["instruction"], "output": row["corrected_sql"]}
            for row in reader
        ]
    jsonl_str = "\n".join(json.dumps(r) for r in records)
    return jsonl_str.encode("utf-8")

# Streamlit UI
st.title("🔭 Natural Language to SDSS SQL Query Tool")

instruction = st.text_area("Enter your astronomy question:")

if st.button("Generate and Run SQL"):
    sql = generate_sql(instruction)
    st.subheader("Generated SQL Query")
    st.code(sql, language="sql")

    save_query_log(instruction, sql)

    with st.spinner("Querying SDSS SkyServer..."):
        df, error = query_sdss(sql)

    if error:
        st.error(f"Error retrieving results: {error}")
        corrected_sql = st.text_area("The generated SQL failed. Enter a corrected SQL query below:")
        if st.button("Submit Corrected Query"):
            df2, err2 = query_sdss(corrected_sql)
            if err2:
                st.error(f"Corrected query also failed: {err2}")
            else:
                save_corrected_query(instruction, corrected_sql)
                st.success("Corrected query succeeded!")
                st.dataframe(df2.head(10))
                csv = df2.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download corrected result as CSV", data=csv, file_name="sdss_corrected_results.csv", mime="text/csv")
    elif df is not None and not df.empty:
        st.success("Query successful! Preview below:")
        st.dataframe(df.head(10))
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download full result as CSV", data=csv, file_name="sdss_query_results.csv", mime="text/csv")
    else:
        st.warning("Query executed but returned no results.")

# Offer JSONL download of corrected training pairs
jsonl_data = convert_corrected_queries_to_jsonl()
if jsonl_data:
    st.download_button("📤 Download corrected queries as JSONL for training", data=jsonl_data, file_name="corrected_training_data.jsonl", mime="application/jsonl")
