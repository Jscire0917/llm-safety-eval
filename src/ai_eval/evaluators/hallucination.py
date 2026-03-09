# src/ai_eval/evaluators/hallucination.py
"""
Hallucination evaluator using retrieval + entailment check.
Generates responses, extracts claims, retrieves documents, and checks support.
"""

from typing import List, Dict, Any
from ai_eval.utils.claims import extract_claims
from ai_eval.models.openai_model import OpenAIModel  # for entailment judge
from ai_eval.retrieval.embedding_retriever import EmbeddingRetriever
from ai_eval.retrieval.knowledge import SAMPLE_DOCUMENTS

class HallucinationEvaluator:
    """
    Measures hallucination by checking if generated claims are supported by retrieved documents.
    Score = fraction of unsupported claims (0.0 = no hallucination, 1.0 = all unsupported).
    """

    def __init__(self, embedding_provider):
        """
        Args:
            embedding_provider: Instance of EmbeddingProvider (OpenAI or SentenceTransformer)
        """
        self.retriever = EmbeddingRetriever(embedding_provider, SAMPLE_DOCUMENTS)
        self.judge_model = OpenAIModel("gpt-4o-mini")  # cheap judge model for entailment

    def evaluate(self, model, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate hallucination on a list of questions/prompts.

        Args:
            model: LLM instance with .generate(prompt) method
            questions: List of dicts with "prompt" key (or plain strings)

        Returns:
            Dict with hallucination_score, per-prompt scores, top failures
        """
        hallucination_rates = []
        per_prompt_scores = []

        for q in questions:
            # Extract prompt string (handle both str and dict)
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
                    # Fast substring match
                    if claim.lower() in doc.lower():
                        supported = True
                        break
                    # Entailment check with judge model
                    judge_prompt = f"Does the document entail the claim?\nDocument: '{doc[:300]}...'\nClaim: '{claim}'\nAnswer only 'yes' or 'no'."
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

        # Top 3 worst hallucination rates
        top_failures = sorted(per_prompt_scores, key=lambda x: x["hallucination_rate"], reverse=True)[:3]

        return {
            "hallucination_score": sum(hallucination_rates) / len(hallucination_rates) if hallucination_rates else 0.0,
            "num_questions": len(questions),
            "per_prompt_scores": per_prompt_scores,
            "top_failures": top_failures
        }