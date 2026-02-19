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
- `/evaluate` endpoint accepts model name, provider, metrics, and optional dataset
- Supports OpenAI (remote) and local models (Ollama-ready)
- Bias: BBQ & CrowS-Pairs inspired prompts + VADER sentiment disparity
- Toxicity: Detoxify (transformer-based multi-class probabilities)
- Hallucination: Retrieval + OpenAI entailment judge + factual knowledge base
- Results saved as JSON + bar chart visualization (`evaluation_results/`)

## Rate Limiting
- The `/evaluate` endpoint is rate-limited to **30 requests per minute per IP** to prevent abuse.

## Cost Tracking
- OpenAI token usage & rough cost estimate included in response (when applicable)

## Model Support: Local + Remote
- The system supports both remote OpenAI models and local Ollama models.

### Remote (OpenAI)
Requires `OPENAI_API_KEY` env var.
```bash
python src/ai_eval/run_benchmark.py --provider openai --model gpt-4o-mini --metrics bias toxicity hallucination

## Local Model Support (Ollama)

- You can run evaluations with local models via Ollama (no OpenAI API key needed for inference).

1. Install Ollama: https://ollama.com
2. Pull a model:
   ```bash
   ollama pull llama3.1:8b
 
## INSTALLATION

```bash
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
- # Terminal One
''bash
python src/ai_eval/run_service.py
- # Terminal Two
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

## RESULTS
- (JSON + plots) will be saved to -> evaluation_results/.

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


