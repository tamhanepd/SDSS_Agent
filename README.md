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

## Fine-tuned Model

The fine-tuned model can be found at [tamhanepd/phi-2-nl-to-sql-sdss](https://huggingface.co/tamhanepd/phi-2-nl-to-sql-sdss).


## 🧪 Test Prompt (Few-shot)
```
User: Show galaxies with redshift < 0.03 and g–r color > 0.4.
Assistant: SELECT * FROM SpecObj AS s JOIN PhotoObj AS p ON s.bestObjID = p.objID WHERE s.class = 'GALAXY' AND s.z < 0.03 AND (p.g - p.r) > 0.4
```
