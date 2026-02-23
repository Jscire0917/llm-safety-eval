from ai_eval.evaluators.bias import BiasEvaluator


class DummyModel:
    def generate(self, prompt):
        return "This is a neutral response."


def test_bias_evaluator_runs():
    evaluator = BiasEvaluator()
    model = DummyModel()
    prompts = [
        {"prompt": "Test prompt", "meta": {"profession": "engineer"}},
        {"prompt": "Another test", "meta": {"profession": "engineer"}},
    ]
    result = evaluator.evaluate(model, prompts)
    assert "bias_score" in result
    assert isinstance(result["bias_score"], (int, float))
    assert result["bias_score"] >= 0.0
