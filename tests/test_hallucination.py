from ai_eval.evaluators.hallucination import HallucinationEvaluator
from ai_eval.retrieval.embedding_base import EmbeddingProvider


class DummyRetriever(EmbeddingProvider):
    def retrieve(self, claim):
        return ["Supporting document."]

    def embed(self, text: str) -> list[float]:
        return [0.0] * 384  # Dummy vector matching common embedding size


class DummyModel:
    def generate(self, prompt):
        return "Factual statement."


def test_hallucination_evaluator_runs():
    retriever = DummyRetriever()
    evaluator = HallucinationEvaluator(retriever)
    model = DummyModel()
    questions = ["What is X?"]
    result = evaluator.evaluate(model, questions)
    assert "hallucination_score" in result
    assert isinstance(result["hallucination_score"], (int, float))
    assert 0.0 <= result["hallucination_score"] <= 1.0
