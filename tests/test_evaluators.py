# tests/test_evaluators.py
import pytest
from ai_eval.evaluators.toxicity import ToxicityEvaluator
from ai_eval.evaluators.bias import BiasEvaluator
from ai_eval.evaluators.hallucination import HallucinationEvaluator
from ai_eval.retrieval.openai_embeddings import OpenAIEmbeddingProvider
from ai_eval.models.openai_model import OpenAIModel
from ai_eval.retrieval.sentence_transformer_provider import SentenceTransformerProvider
from ai_eval.utils.costs import CostTracker

class DummyModel:
    def generate(self, prompt):
        return "This is a safe, neutral, factual response."


@pytest.fixture
def dummy_model():
    return DummyModel()


def test_toxicity_evaluator_smoke(dummy_model):
    evaluator = ToxicityEvaluator(model_name='original-small')  # fast model for tests
    prompts = [{"prompt": "This is a safe sentence."}]
    result = evaluator.evaluate(dummy_model, prompts)
    assert "toxicity_score" in result
    assert 0.0 <= result["toxicity_score"] <= 1.0
    assert "num_prompts_evaluated" in result
    assert result["num_prompts_evaluated"] == 1


def test_bias_evaluator_smoke(dummy_model):
    evaluator = BiasEvaluator()
    prompts = [
        {"prompt": "Test engineer", "meta": {"profession": "engineer"}},
        {"prompt": "Test nurse", "meta": {"profession": "nurse"}},
    ]
    result = evaluator.evaluate(dummy_model, prompts)
    assert "bias_score" in result
    assert result["bias_score"] >= 0.0
    assert "num_prompts" in result


def test_hallucination_evaluator_openai_smoke():
    provider = OpenAIEmbeddingProvider("text-embedding-3-small")
    evaluator = HallucinationEvaluator(provider)
    model = DummyModel()
    questions = ["What is the capital of France?"]
    result = evaluator.evaluate(model, questions)
    assert "hallucination_score" in result
    assert 0.0 <= result["hallucination_score"] <= 1.0


def test_hallucination_evaluator_local_smoke():
    provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
    evaluator = HallucinationEvaluator(provider)
    model = DummyModel()
    questions = ["What is the capital of France?"]
    result = evaluator.evaluate(model, questions)
    assert "hallucination_score" in result
    assert 0.0 <= result["hallucination_score"] <= 1.0
    
def test_cost_tracker():
    tracker = CostTracker()
    tracker.add_usage(prompt="Hello " * 100, completion="world " * 50)
    summary = tracker.get_summary()
    assert summary["total_prompt_tokens"] > 0
    assert summary["total_completion_tokens"] > 0
    assert summary["estimated_cost_usd"] >= 0


@pytest.mark.asyncio
async def test_async_evaluation():
    # Simple smoke test for async gather
    from ai_eval.service.api import run_evaluator  # if you extracted it
    # ... mock model and evaluators ...
    pass