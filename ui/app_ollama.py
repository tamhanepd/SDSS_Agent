import streamlit as st
import requests
import pandas as pd
import yaml
from io import StringIO

st.title("🔭 SDSS Natural Language to SQL Chatbot (Ollama + Live Query)")

# Load config
with open("configs/app_config.yaml") as f:
    cfg = yaml.safe_load(f)

OLLAMA_URL = cfg["ollama_url"]
OLLAMA_MODEL = cfg["ollama_model_name"]
SKYSERVER_URL = cfg["skyserver_url"]
RESULT_FORMAT = cfg.get("sql_result_format", "csv")

def query_ollama(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    })
    if response.ok:
        return response.json()["response"]
    else:
        return "Error communicating with Ollama server."

def query_sdss(sql_query):
    params = {
        "cmd": sql_query,
        "format": RESULT_FORMAT
    }
    response = requests.get(SKYSERVER_URL, params=params)
    if response.status_code == 200:
        df = pd.read_csv(StringIO(response.text))
        return df
    else:
        raise RuntimeError(f"SDSS query failed: {response.status_code}")

user_input = st.text_input("Ask a question about SDSS data:")
if user_input:
    full_prompt = f"User: {user_input}\nAssistant:"
    sql_output = query_ollama(full_prompt)
    st.code(sql_output.strip(), language="sql")

    try:
        with st.spinner("Querying SDSS SkyServer..."):
            df = query_sdss(sql_output.strip())
        st.success("Query successful! Showing results:")
        st.dataframe(df)
        csv = df.to_csv(index=False)
        st.download_button("Download Results as CSV", csv, "sdss_results.csv")
    except Exception as e:
        st.error(f"Failed to query SDSS: {e}")