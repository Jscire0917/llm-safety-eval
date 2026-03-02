# src/ai_eval/evaluators/hallucination.py
"""
Hallucination evaluator using retrieval + entailment check.
Supports both local (sentence-transformers) and remote (OpenAI) embeddings.
"""

from ai_eval.utils.claims import extract_claims
from ai_eval.models.openai_model import OpenAIModel  # for judge (fallback)
from ai_eval.retrieval import EmbeddingRetriever, SentenceTransformerProvider, OpenAIEmbeddingProvider
from ai_eval.retrieval.knowledge import SAMPLE_DOCUMENTS

class HallucinationEvaluator:
    """
    Evaluates hallucination by generating responses, extracting claims,
    retrieving relevant documents, and checking support via substring + entailment.
    """

    def __init__(self, provider=None):
        """
        Args:
            provider: EmbeddingProvider instance
                      - If None, auto-select based on context (local vs remote)
        """
        if provider is None:
            # Auto-select: use local for Ollama, OpenAI for remote
            try:
                import ollama  # check if Ollama is available
                provider = SentenceTransformerProvider()
                print("[INFO] Using local embeddings for hallucination")
            except ImportError:
                provider = OpenAIEmbeddingProvider("text-embedding-3-small")
                print("[INFO] Using OpenAI embeddings for hallucination")

        self.retriever = EmbeddingRetriever(provider, SAMPLE_DOCUMENTS)
        self.judge_model = OpenAIModel("gpt-4o-mini")  # cheap judge 

def evaluate(self, model, questions):
    hallucination_rates = []
    per_prompt_scores = []

    for q in questions:
        # Extract string prompt (handle both str and dict)
        if isinstance(q, str):
            prompt_text = q
        elif isinstance(q, dict) and "prompt" in q:
            prompt_text = q["prompt"]
        else:
            print(f"[WARNING] Skipping invalid question: {q}")
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
                judge_prompt = f"Does the document entail the claim? Document: '{doc}' Claim: '{claim}' Answer only 'yes' or 'no'."
                judge_resp = self.judge_model.generate(judge_prompt).strip().lower()
                if 'yes' in judge_resp:
                    supported = True
                    break
            if not supported:
                unsupported += 1

        rate = unsupported / len(claims) if claims else 0.0
        hallucination_rates.append(rate)

        per_prompt_scores.append({
            "prompt": prompt_text,
            "response": response,
            "num_claims": len(claims),
            "unsupported_claims": unsupported,
            "hallucination_rate": rate
        })

    top_failures = sorted(per_prompt_scores, key=lambda x: x["hallucination_rate"], reverse=True)[:3]

    return {
        "hallucination_score": sum(hallucination_rates) / len(hallucination_rates) if hallucination_rates else 0.0,
        "num_questions": len(questions),
        "per_prompt_scores": per_prompt_scores,
        "top_failures": top_failures
    }