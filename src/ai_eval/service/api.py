from fastapi import FastAPI, Depends, HTTPException, Header
from openai import api_key
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import os
import numpy as np
import asyncio
from fastapi.responses import HTMLResponse
from ai_eval.models.factory import get_model
from ai_eval.evaluators.bias import BiasEvaluator
from ai_eval.evaluators.toxicity import ToxicityEvaluator
from ai_eval.evaluators.hallucination import HallucinationEvaluator
from ai_eval.retrieval.openai_embeddings import OpenAIEmbeddingProvider
from ai_eval.retrieval.sentence_transformer_provider import SentenceTransformerProvider
from ai_eval.utils.reporting import save_results, generate_visual_report
from .auth import VALID_KEYS, authenticate
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from ai_eval.utils.costs import CostTracker
from ai_eval.utils.logging import logger
from fastapi import WebSocket 

# Main FastAPI application
# Provides /health (liveness check) and /evaluate (runs LLM safety metrics)
app = FastAPI(
    title="Agnostic LLM Evaluation Service",
    description="Model-agnostic evaluation of bias, toxicity, and hallucination",
    version="0.1.0"
)

# Initialize limiter (30 requests per minute per IP)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class EvalRequest(BaseModel):
    model_name: str
    provider: str = "openai"
    prompts: List[Dict[str, Any]] | None = None
    metrics: List[str] | None = None
    use_dataset: str = "bbq"  # "bbq" or "crows"
    custom_prompts: Optional[List[Dict[str, Any]]] = None  # for custom prompts
    class Config:
        json_schema_extra = {  # Example for documentation
            "example": {
                "model_name": "gpt-4o-mini",
                "provider": "openai",
                "prompts": [
                    {
                        "prompt": "Describe a typical software engineer.",
                        "meta": {"profession": "engineer", "gender": "male"}
                    },
                    {
                        "prompt": "Describe a typical nurse.",
                        "meta": {"profession": "nurse", "gender": "female"}
                    }
                ],
                "metrics": ["bias", "toxicity"],
                "use_dataset": "bbq"  # Optional dataset name
            }
        }

class EvalResponse(BaseModel):
    results: Dict[str, Any]
    duration_seconds: float
    model_used: str
    evaluator_versions: Dict[str, str] 
    
@app.post("/health")
def health():
    return {"status": "ok"}

@app.post("/evaluate", response_model=EvalResponse)
@limiter.limit("30/minute")
async def evaluate(
    request: Request, 
    eval_request: EvalRequest, 
    api_key: str = Depends(authenticate),
):
    start_time = time.time()
    cost_tracker = CostTracker()
    
    try:
        model = get_model(eval_request.provider, eval_request.model_name)

    # Add cost tracking wrapper only for OpenAI provider
        if eval_request.provider.lower() in ("openai", "azure"):
            original_generate = model.generate

        def tracked_generate(prompt: str) -> str:
            response = original_generate(prompt)
            cost_tracker.add_usage(prompt=prompt, completion=response)
            return response

        model.generate = tracked_generate
    
    except Exception as e:
    # Log the error for debugging
        logger.error(f"Model loading failed for {eval_request.provider}/{eval_request.model_name}: {str(e)}")
        raise HTTPException(
        status_code=400,
        detail=f"Model loading failed: {str(e)}"
    )
    
    # Build evaluators list
    metrics = eval_request.metrics or ["bias", "toxicity", "hallucination"]
    evaluators_list: List[Any] = []

    if "bias" in metrics:
        evaluators_list.append(BiasEvaluator())
    if "toxicity" in metrics:
        evaluators_list.append(ToxicityEvaluator(model_name='original'))
    if "hallucination" in metrics:
        # Default to OpenAI embeddings (requires OPENAI_API_KEY)
        embedding_provider = OpenAIEmbeddingProvider("text-embedding-3-small")

        # Override to local embeddings for Ollama (fully offline)
        if eval_request.provider.lower() == "ollama":
            embedding_provider = SentenceTransformerProvider("all-MiniLM-L6-v2")

        evaluators_list.append(HallucinationEvaluator(embedding_provider))

    if not evaluators_list:
        raise HTTPException(status_code=400, detail="No valid metrics selected")

    # Async parallel evaluation
    async def run_evaluator(evaluator, model, data, **kwargs):
        name = evaluator.__class__.__name__
        try:
            result = await asyncio.to_thread(evaluator.evaluate, model, data, **kwargs)
            logger.info(f"Completed evaluator: {name} successfully")
            return name, result
        except Exception as e:
            logger.error(f"Evaluator {name} failed: {str(e)}")
            return name, {"error": str(e)}

    tasks = []
    for evaluator in evaluators_list:
        name = evaluator.__class__.__name__

        # Per-evaluator kwargs (bias uses use_dataset)
        eval_kwargs: Dict[str, Any] = {}
        if name == "BiasEvaluator" and eval_request.use_dataset:
            eval_kwargs["use_dataset"] = eval_request.use_dataset

        # Determine data source (custom prompts > dataset > fallback defaults)
        if eval_request.prompts:
            data = eval_request.prompts
        else:
            # Dataset fallback (bias supports it)
            if name == "BiasEvaluator":
                if eval_request.use_dataset == "bbq":
                    from ai_eval.datasets.datasets import BBQ_SUBSET
                    data = BBQ_SUBSET
                elif eval_request.use_dataset == "crows":
                    from ai_eval.datasets.datasets import CROWS_PAIRS_SUBSET
                    data = CROWS_PAIRS_SUBSET
                elif eval_request.use_dataset == "realtoxicity":
                    from ai_eval.datasets.extra_datasets import REAL_TOXICITY_PROMPTS_SUBSET
                    data = REAL_TOXICITY_PROMPTS_SUBSET
                elif eval_request.use_dataset == "truthfulqa":
                    from ai_eval.datasets.extra_datasets import TRUTHFULQA_SUBSET
                    data = TRUTHFULQA_SUBSET
                else:
                    raise HTTPException(status_code=400, detail="Invalid use_dataset for bias")
            # Toxicity fallback: challenging/toxic prompts to test properly
            elif name == "ToxicityEvaluator":
                data = REAL_TOXICITY_PROMPTS_SUBSET  # Use the RealToxicityPrompts subset as default for toxicity testing
            # Hallucination fallback: tricky questions to induce hallucinations
            elif name == "HallucinationEvaluator":
                data = TRUTHFULQA_SUBSET  # Use TruthfulQA subset as default for hallucination testing
            else:
                raise HTTPException(status_code=400, detail=f"{name} requires prompts or supported dataset")

        tasks.append(run_evaluator(evaluator, model, data, **eval_kwargs))
    # Run evaluators in parallel
    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect results
    results: Dict[str, Any] = {}
    for name, res in results_list:
        if isinstance(res, Exception):
            logger.error(f"Evaluator {name} failed during gather: {str(res)}")
            results[name] = {"error": str(res)}
        else:
            logger.info(f"Evaluator {name} completed (from gather)")
            results[name] = res

    # Convert NumPy types to Python natives for Pydantic/FastAPI serialization
    def convert_np_for_pydantic(obj: Any) -> Any:
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return tuple(convert_np_for_pydantic(item) for item in obj)
        if isinstance(obj, list):
            return [convert_np_for_pydantic(item) for item in obj]
        if isinstance(obj, dict):
            return {k: convert_np_for_pydantic(v) for k, v in obj.items()}
        return obj

    clean_results = convert_np_for_pydantic(results)

    # Persist results and generate visual report
    save_results(clean_results, output_dir="evaluation_results")
    generate_visual_report(clean_results, output_dir="evaluation_results")
    
    # Save results to SQLite for leaderboard
    from ai_eval.db.db_utils import save_evaluation
    save_evaluation(eval_request.model_name, eval_request.provider, results)
    
    # Add cost summary to results
    results["cost_summary"] = cost_tracker.get_summary()

    return EvalResponse(
        results=clean_results,
        duration_seconds=round(time.time() - start_time, 2),
        model_used=f"{eval_request.provider}/{eval_request.model_name}",
        evaluator_versions={"bias": "v1", "toxicity": "detoxify", "hallucination": "embedding-v1"}
    )
@app.get("/leaderboard")
def leaderboard():
    from ai_eval.db.db_utils import get_leaderboard
    rows = get_leaderboard()
    leaderboard_html = "<h1>LLM Safety Leaderboard</h1><table><tr><th>Model</th><th>Provider</th><th>Avg Score (lower is better)</th></tr>"
    for row in rows:
        leaderboard_html += f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{round(row[2], 4)}</td></tr>"
    leaderboard_html += "</table>"
    return HTMLResponse(content=leaderboard_html)
                                                                          
@app.websocket("/ws/evaluate")
async def websocket_evaluate(websocket: WebSocket):
    evaluators_list: List[Any] = []
    # Extract api_key from query params (or header if you prefer)
    api_key = websocket.query_params.get("api_key")
    if not api_key:
        await websocket.close(code=1008, reason="API key required")
        return

    try:
        authenticate(api_key) 
    except HTTPException as exc:
        await websocket.close(code=1008, reason=exc.detail)
        return

    await websocket.accept()
    await websocket.send_text("Evaluation started - connected")

    try:
        data = await websocket.receive_json()
        eval_request = EvalRequest(**data)

        await websocket.send_text("Loading model...")
        model = get_model(eval_request.provider, eval_request.model_name)

        metrics = eval_request.metrics or ["bias", "toxicity", "hallucination"]
        await websocket.send_text(f"Preparing {len(metrics)} evaluators...")

        
        results = {}
        for evaluator in evaluators_list:
            name = evaluator.__class__.__name__
            await websocket.send_text(f"Running {name}...")
            # Run evaluator 
            result = evaluator.evaluate(model, data)  # Adjust to your data source
            results[name] = result
            await websocket.send_text(f"{name} complete")

        await websocket.send_text("Saving results and generating report...")
        save_results(results, output_dir="evaluation_results")
        generate_visual_report(results, output_dir="evaluation_results")

        await websocket.send_json({"results": results, "status": "complete"})
    except Exception as e:
        await websocket.send_text(f"Error: {str(e)}")
    finally:
        await websocket.close()