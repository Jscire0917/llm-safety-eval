# src/ai_eval/dashboard.py
"""
Streamlit dashboard for interactive LLM safety evaluation.
Runs on http://localhost:8501
"""

import streamlit as st
import requests
import json

st.title("LLM Safety Evaluation Dashboard")

with st.sidebar:
    model_name = st.text_input("Model Name", "gpt-4o-mini")
    provider = st.selectbox("Provider", ["openai", "ollama"])
    use_dataset = st.selectbox("Dataset", ["bbq", "crows", "realtoxicity", "truthfulqa"])
    metrics = st.multiselect("Metrics", ["bias", "toxicity", "hallucination"], default=["bias", "toxicity", "hallucination"])
    api_key = st.text_input("API Key", "sk-dummy-local", type="password")

if st.button("Run Evaluation"):
    payload = {
        "model_name": model_name,
        "provider": provider,
        "metrics": metrics,
        "use_dataset": use_dataset
    }
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
    response = requests.post("http://localhost:8000/evaluate", headers=headers, json=payload)
    if response.status_code == 200:
        results = response.json()
        st.json(results)
        st.image("evaluation_results/scores_bar.png", caption="Scores Bar Plot")
    else:
        st.error(f"Error: {response.status_code} - {response.text}")

st.markdown("Server must be running: `python src/ai_eval/run_service.py`")