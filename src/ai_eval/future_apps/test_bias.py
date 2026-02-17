from ai_eval.evaluators.bias import BiasEvaluator

class DummyModel:
    def generate(self, prompt):
        return "This is a good response."

def test_bias_evaluator_runs():
    evaluator = BiasEvaluator()
    model = DummyModel()

    prompts = [
        {"prompt": "Test", "meta": {"profession": "engineer"}},
        {"prompt": "Test", "meta": {"profession": "engineer"}},
    ]

    result = evaluator.evaluate(model, prompts)
    assert "bias_score" in result
