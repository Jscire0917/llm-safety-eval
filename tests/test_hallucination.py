from ai_eval.evaluators.hallucination import HallucinationEvaluator

class DummyRetriever:
    def retrieve(self, claim):
        return ["This document supports the claim."]

class DummyModel:
    def generate(self, prompt):
        return "This is a factual statement."

def test_hallucination_evaluator_runs():
    evaluator = HallucinationEvaluator(DummyRetriever())
    model = DummyModel()

    result = evaluator.evaluate(model, ["What is X?"])
    assert "hallucination_score" in result
