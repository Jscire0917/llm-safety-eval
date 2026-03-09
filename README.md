# LLM-Safety-Eval
<p align="center">
  <a href="https://github.com/Jscire0917/llm-safety-eval">
    <img src="https://img.shields.io/github/stars/Jscire0917/llm-safety-eval?style=for-the-badge&logo=github&color=green" alt="GitHub stars" />
  </a>
  <a href="https://github.com/Jscire0917/llm-safety-eval">
    <img src="https://img.shields.io/github/forks/Jscire0917/llm-safety-eval?style=for-the-badge&logo=github&color=blue" alt="GitHub forks" />
  </a>
  <a href="https://github.com/Jscire0917/llm-safety-eval/issues">
    <img src="https://img.shields.io/github/issues/Jscire0917/llm-safety-eval?style=for-the-badge&logo=github" alt="GitHub issues" />
  </a>
  <a href="https://github.com/Jscire0917/llm-safety-eval/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Jscire0917/llm-safety-eval?style=for-the-badge&color=purple" alt="License" />
  </a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python version" />
  <img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
</p>

Model-agnostic FastAPI service to evaluate large language models for **bias**, **toxicity**, and **hallucination**.

## Features
- '/evaluate' endpoint accepts model name, provider, metrics, and optional dataset
## WebSocket for Real-Time Progress
- Use '/ws/evaluate' for streaming updates.

- Example Python client:
'''python
import websocket
import json

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
## Features Continued... 
- Supports OpenAI (remote) and local models (Ollama-ready)
- Bias: BBQ & CrowS-Pairs inspired prompts + VADER sentiment disparity
- Toxicity: Detoxify (transformer-based multi-class probabilities)
- Hallucination: Retrieval + OpenAI entailment judge + factual knowledge base
- Results saved as JSON + bar chart visualization ('evaluation_results/')
## Leaderboard
- Results are saved to SQLite ('src/ai_eval/db/results.db'). View ranked models at http://localhost:8000/leaderboard (simple HTML table).
## Rate Limiting
- The '/evaluate' endpoint is rate-limited to **30 requests per minute per IP** to prevent abuse.
## Custom Prompts
- Use "custom_prompts" in POST body for your own prompts.
## Cost Tracking
- OpenAI token usage & rough cost estimate included in response (when applicable)


### To Get Toxicity & Hallucination Results Use This In Terminal⬇(With your own prompts)
python src/ai_eval/run_benchmark.py \
--metrics bias toxicity hallucination \
--prompts '[{"prompt": "This is a test sentence."}, {"prompt": "The weather is nice."}]'

Note: Toxicity and hallucination require prompts or their supported datasets (realtoxicity, truthfulqa) for meaningful evaluation. If no prompts are provided, they use relevant defaults (RealToxicityPrompts for toxicity, TruthfulQA for hallucination). Provide custom prompts for serious benchmarking.


## Model Support: Local + Remote
- The system supports both remote OpenAI models and local Ollama models.

### Remote (OpenAI)
### Requires "OPENAI_API_KEY" env var.
''bash
python src/ai_eval/run_benchmark.py --provider openai --model gpt-4o-mini --metrics bias toxicity hallucination

## Local Model Support (Ollama)
- You can run evaluations with local models via Ollama (no OpenAI API key needed for inference).

1. Install Ollama: https://ollama.com
2. Pull a model:
   '''bash
   ollama pull llama3.1:8b
 
## Local Model Support – MLX (Apple Silicon Optimized)
- For M1/M2/M3/M4 Macs, MLX provides extremely fast inference using the Neural Engine + GPU cores.

1. Install MLX:
   '''bash
   pip install mlx mlx-lm

## INSTALLATION
'''bash
git clone https://github.com/Jscire0917/llm-safety-eval.git
cd llm-safety-eval
python -m venv .venv
source .venv/bin/activate
pip install -e .

-- Alternative install (without editable mode):
pip install -r requirements.txt

## Citation
- Inspired by BBQ [https://arxiv.org/abs/2110.08193] and CrowS-Pairs [https://arxiv.org/abs/2010.00133]

## SET UP API_KEYS. You need two environment variables. 
## Required for OpenAI models & embeddings
export OPENAI_API_KEY="sk-proj-your-real-key-here"

## For service authentication (can be any string, e.g. dummy for local)
export API_KEYS="sk-dummy-local"

## RUNNING: -> Two terminals(recommended for development)
- ## Terminal One
''bash
python src/ai_eval/run_service.py
- ## Terminal Two
''bash
python src/ai_eval/run_benchmark.py --dataset realtoxicity --metrics toxicity
python src/ai_eval/run_benchmark.py --dataset truthfulqa --metrics hallucination

## RUNNING: Option two
- # One Command(combined run)
''bash
python src/ai_eval/run_service.py & \
until curl -s -f -X POST http://127.0.0.1:8000/health -d '' >/dev/null 2>&1; do \
    echo "Waiting for server..."; sleep 1; \
done && \
python src/ai_eval/run_benchmark.py --dataset bbq --metrics bias toxicity hallucination && \
fg

## Logging
- Errors and info logged to console + evaluation_results/logs.txt (loguru).

## RESULTS
- (JSON + plots) will be saved to -> evaluation_results/.
## Detailed Results
- Results include per-prompt scores and top 3 failures per metric.

## Streamlit Dashboard
- Interactive UI for running evaluations:
'''bash
streamlit run src/ai_eval/dashboard.py

## DOCKER(Optional)
''bash 
docker build -t llm-safety-eval .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-proj-... \
  -e API_KEYS=sk-dummy-local \
  llm-safety-eval

- Interactive docs: http://localhost:8000/docs

## TESTING
- Run the test suite:
''bash
pytest tests/

## License 
-- MIT


