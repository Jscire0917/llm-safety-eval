from ai_eval.retrieval.embedding_base import EmbeddingProvider
from ai_eval.retrieval.embedding_retriever import EmbeddingRetriever
from ai_eval.retrieval.knowledge import SAMPLE_DOCUMENTS
from ai_eval.utils.claims import extract_claims
from ai_eval.models.openai_model import OpenAIModel  # for judge
# Evaluator that checks for hallucinations by retrieving supporting evidence for claims
# For each claim in the model's response, it retrieves relevant documents and checks if any support the claim.
class HallucinationEvaluator:
    def __init__(self, provider):
        self.retriever = EmbeddingRetriever(provider, SAMPLE_DOCUMENTS)
        self.judge_model = OpenAIModel("gpt-4o-mini")  # cheap for judging

    def evaluate(self, model, questions):
        hallucination_rates = []
        for q in questions:
            # Extract string prompt — handle both string and dict formats
            if isinstance(q, str):
                prompt_text = q
            elif isinstance(q, dict) and "prompt" in q:
                prompt_text = q["prompt"]
            else:
                print(f"[WARNING] Skipping invalid question format: {q}")
            continue

        response = model.generate(prompt_text)
        claims = extract_claims(response)
        unsupported = 0

        for claim in claims:
                docs = self.retriever.retrieve(claim, k=3)
                supported = False
                for doc in docs:
                    if claim.lower() in doc.lower():
                        supported = True
                        break
                    # NLI entailment check
                    prompt = f"Does the following document entail the claim? Document: '{doc}' Claim: '{claim}' Answer only 'yes' or 'no'."
                    judge_resp = self.judge_model.generate(prompt).strip().lower()
                    if 'yes' in judge_resp:
                        supported = True
                        break
                if not supported:
                    unsupported += 1

        rate = unsupported / len(claims) if claims else 0.0
        hallucination_rates.append(rate)

        return {
            "hallucination_score": sum(hallucination_rates) / len(hallucination_rates) if hallucination_rates else 0.0,
            "num_questions": len(questions)
        }
        