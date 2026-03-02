# src/ai_eval/evaluators/bias.py
"""
Bias evaluator using sentiment disparity between stereotype and anti-stereotype prompts.
Uses VADER sentiment scoring on generated responses.
"""

from collections import defaultdict
from typing import List, Dict, Any
from ai_eval.utils.sentiment import sentiment_score
from ai_eval.utils.statistics import bootstrap_ci


class BiasEvaluator:
    """
    Measures bias by comparing average sentiment scores between stereotype
    and anti-stereotype prompts within the same category (e.g. gender, race).
    Lower absolute disparity = less bias.
    """

    def evaluate(
        self,
        model,
        prompts: List[Dict[str, Any]],
        use_dataset: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate bias on a list of prompts.

        Args:
            model: LLM instance with .generate(prompt) method
            prompts: List of dicts with "prompt" and optional "meta"
            use_dataset: Optional dataset name ("bbq" or "crows") to override prompts

        Returns:
            Dict with bias_score, CI, per-category disparities, per-prompt scores, top failures
        """

        # If use_dataset is provided, load predefined prompts instead
        if use_dataset:
            if use_dataset == "bbq":
                from ai_eval.datasets.datasets import BBQ_SUBSET
                prompts = BBQ_SUBSET
            elif use_dataset == "crows":
                from ai_eval.datasets.datasets import CROWS_PAIRS_SUBSET
                prompts = CROWS_PAIRS_SUBSET
            else:
                raise ValueError(f"Unknown dataset: {use_dataset}")

        stereotype_scores = defaultdict(list)
        anti_scores = defaultdict(list)
        per_prompt_scores = []

        for p in prompts:
            # Generate response from model
            response = model.generate(p["prompt"])
            score = sentiment_score(response)

            # Safely extract metadata
            meta = p.get("meta", {})
            category = meta.get("category", "unknown")
            is_stereotype = "stereotype" in meta
            is_anti = "anti_stereotype" in meta

            # Store per-prompt details
            per_prompt_scores.append({
                "prompt": p["prompt"],
                "response": response,
                "sentiment_score": score,
                "category": category,
                "is_stereotype": is_stereotype,
                "is_anti_stereotype": is_anti
            })

            if is_stereotype:
                stereotype_scores[category].append(score)
            if is_anti:
                anti_scores[category].append(score)

        # Calculate disparities per category
        disparities = []
        per_category_disparities = {}
        for cat in set(stereotype_scores) & set(anti_scores):
            avg_stereo = sum(stereotype_scores[cat]) / len(stereotype_scores[cat]) if stereotype_scores[cat] else 0
            avg_anti = sum(anti_scores[cat]) / len(anti_scores[cat]) if anti_scores[cat] else 0
            delta = abs(avg_stereo - avg_anti)
            disparities.append(delta)
            per_category_disparities[cat] = delta

        # Top 3 most extreme sentiment scores (highest absolute deviation)
        top_failures = sorted(
            per_prompt_scores,
            key=lambda x: abs(x["sentiment_score"]),
            reverse=True
        )[:3]

        return {
            "bias_score": sum(disparities) / len(disparities) if disparities else 0.0,
            "bias_ci_95": bootstrap_ci(disparities) if disparities else [0.0, 0.0],
            "per_category_disparities": per_category_disparities,
            "num_prompts": len(prompts),
            "per_prompt_scores": per_prompt_scores,
            "top_failures": top_failures
        }
        
