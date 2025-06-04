# SDSS Natural Language to SQL Chatbot

This project builds a lightweight chatbot that translates natural language astronomy queries into SQL queries compatible with the Sloan Digital Sky Survey (SDSS) database. It uses small open-source language models like `phi-2` or `mistral` and supports fine-tuning using LoRA on a small dataset.

## 🚀 Features
- Natural language → SDSS SQL translation
- LoRA fine-tuning on custom instruction data
- Lightweight enough for local CPU-based training/inference
- Optional Streamlit interface for chat-style querying

## 🗂️ Project Structure
- `data/`: Raw and processed training datasets
- `models/`: Base models, LoRA weights, and merged models
- `scripts/`: Training, inference, preprocessing scripts
- `ui/`: Streamlit-based chatbot interface
- `configs/`: LoRA and training configuration files

## 📦 Requirements
See `requirements.txt` for all necessary packages.

## 📈 Training
```bash
python scripts/train_phi2_lora.py
```

## 💬 Inference
```bash
python scripts/run_inference.py
```

## 🧪 Test Prompt (Few-shot)
```
User: Show galaxies with redshift < 0.03 and g–r color > 0.4.
Assistant: SELECT * FROM SpecObj AS s JOIN PhotoObj AS p ON s.bestObjID = p.objID WHERE s.class = 'GALAXY' AND s.z < 0.03 AND (p.g - p.r) > 0.4
```

## 🔓 License
MIT

---

*Built with ❤️ for astronomy + LLMs*

---

## 🛠️ Manual Steps Required

Although most of the pipeline is automated, there are a few steps that need to be done manually for Ollama deployment:

### 1. Quantize Merged Model to GGUF
After running `scripts/quantize_to_gguf.py`, convert the saved model in `models/ollama_ready/` to `.gguf` format using one of:

#### Option A: Using `llama.cpp`
```bash
cd llama.cpp
python3 convert.py --outfile phi2_sdss.Q4_K_M.gguf --outtype q4_k_m --vocab-type bpe /path/to/app/models/ollama_ready/
```

#### Option B: Using `text-generation-webui`
- Load the model from `models/ollama_ready/`
- Use the "Save as GGUF" option in the GUI

### 2. Create Ollama Model
Once you have `phi2_sdss.Q4_K_M.gguf` and your `Modelfile` in `models/ollama_ready/`, build the Ollama model:

```bash
cd app/models/ollama_ready/
ollama create sdss-phi2 -f Modelfile
```

Then run it:

```bash
ollama run sdss-phi2
```

You can now use `ui/app_ollama.py` to query your custom model via a local API.
