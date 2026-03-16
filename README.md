# LLM Safety Eval

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/github/license/Jscire0917/llm-safety-eval?style=for-the-badge&color=purple" alt="License" />
  <img src="https://img.shields.io/github/stars/Jscire0917/llm-safety-eval?style=for-the-badge&logo=github&color=green" alt="Stars" />
  <img src="https://img.shields.io/github/issues/Jscire0917/llm-safety-eval?style=for-the-badge&logo=github" alt="Issues" />
</p>

**Model-agnostic FastAPI service** to evaluate large language models for **bias**, **toxicity**, and **hallucination**. Supports OpenAI, Ollama, MLX (Apple Silicon), and more.

### Features at a Glance
- 🧠 Bias detection using BBQ/CrowS-Pairs + VADER sentiment disparity
- 🔥 Toxicity scoring with Detoxify (multi-class probabilities + CI)
- 🤖 Hallucination detection using retrieval + entailment judge
- 📊 Per-prompt breakdown + top 3 failures per metric
- 📈 Leaderboard with model comparison table
- ⚡ Real-time WebSocket progress updates
- 🔄 Batch evaluation (run multiple models at once)
- 💡 Mitigation suggestions for high scores
- 💾 SQLite storage + cost tracking (OpenAI)
- 🖥️ Streamlit dashboard (optional)

## Quick Start

1. **Clone & Install**

'''bash
  git clone https://github.com/Jscire0917/llm-safety-eval.git
  cd llm-safety-eval
  python -m venv .venv
  source .venv/bin/activate
  pip install -e .


## Set API keys
'''Bash
  export OPENAI_API_KEY="sk-proj-your-real-key"
  export API_KEYS="sk-dummy-local"  # any string for local auth


## Run server + benchmark in one command
'''Bash
  python src/ai_eval/run_service.py & \
  until curl -s -f -X POST http://127.0.0.1:8000/health -d '' >/dev/null 2>&1; do \
    sleep 1; \
  done && \
  python src/ai_eval/run_benchmark.py --dataset bbq --metrics bias toxicity hallucination && \fg

**Results (JSON + bar chart) saved to evaluation_results/.**


## Interactive docs
- http://localhost:8000/docs


## Supported Providers

Provider |  Type |      Setup Command        |                Example Model                 |            Notes             |    
_________|_______|___________________________|______________________________________________|______________________________|
- Openai | Remote| export OPENAI_API_KEY=... | gpt-4o-mini                                  |  Requires API key            |  
- Ollama | Local | ollama pull llama3.1:8b   | llama3.1:8b                                  |  Fast on CPU/Neural Engine   |
- Mlx    | Local | pip install mlx mlx-lm,   | mlx-community/Meta-Llama-3.1-8B-Instruct-4bit|  Fastest on M1/M2/M3/M4 Macs |
           

## Endpoints

Method |     Endpoint          |            Description                    |
_______|_______________________|___________________________________________| 
POST   | /evaluate             | Run single evaluation                     |
POST   |/batch/evaluate.       | Run multiple evaluations (returns job ID) |
GET    |/batch/status/{job_id} | Check batch job status & results          |
GET    |/leaderboard           | View ranked models + comparison table     |
WS     |/ws/evaluate           | Real-time progress updates (WebSocket)    |



## Advanced Usage
- Batch Evaluation
'''Bash
  curl -X POST http://localhost:8000/batch/evaluate \
  -H "api-key: sk-dummy-local" \
  -H "Content-Type: application/json" \
  -d '{
    "evaluations": [
      {"model_name": "gpt-4o-mini", "provider": "openai", "metrics": ["bias"]},
      {"model_name": "llama3.1:8b", "provider": "ollama", "metrics": ["toxicity"]}
    ]
  }'

**Check status: GET /batch/status/{job_id}**



## Real-Time Progress (WebSocket)
Python
import websocket, json

ws = websocket.create_connection("ws://localhost:8000/ws/evaluate?api_key=sk-dummy-local")
ws.send(json.dumps({
    "model_name": "gpt-4o-mini",
    "provider": "openai",
    "metrics": ["bias", "toxicity"]
}))
while True:
    msg = ws.recv()
    print(msg)
    if "complete" in msg:
        break
ws.close()



## Leaderboard
- Visit: http://localhost:8000/leaderboard
- Shows ranked models + per-metric comparison table (bias, toxicity, hallucination averages).


## Streamlit Dashboard
'''Bash
streamlit run src/ai_eval/dashboard.py

- Interactive UI to select model, metrics, dataset, and see results live.


## Custom Prompts
'''JSON
{
"model_name": "gpt-4o-mini",
"provider": "openai",
"metrics": ["toxicity"],
"prompts": [
  {"prompt": "Your custom test prompt here"}  
  ]
}


## Logging & Results
- Logs: evaluation_results/logs.txt + console
- Results: JSON + bar charts in evaluation_results/



## Installation & Setup
- Requirements: Python 3.10+
- Dependencies: FastAPI, Uvicorn, OpenAI, Detoxify, Sentence-Transformers, MLX (for Apple Silicon), etc.
- API Keys:
- OPENAI_API_KEY for remote models/embeddings
- API_KEYS for service authentication (e.g. sk-dummy-local)


## Testing
'''Bash
pytest tests/
- Includes smoke tests for API and evaluators.


## License
- MIT License


## Notes:
- Toxicity may require initial internet connection to download model weights (Detoxify).
- Hallucination uses OpenAI embeddings by default (can be local with Ollama + Sentence-Transformers).
- Leaderboard requires running multiple models to see meaningful rankings.

## Contributions welcome! See CONTRIBUTING.md ## 


