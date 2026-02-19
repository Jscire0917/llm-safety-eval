# src/ai_eval/utils/costs.py
"""
Simple OpenAI token usage tracker & estimator.
Tracks real usage when available, falls back to rough estimate otherwise.
"""

from typing import Dict, Optional


class CostTracker:
    """
    Tracks OpenAI API token usage and estimated cost.
    Uses real usage when the response provides it, otherwise rough estimate.
    """

    def __init__(self):
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.estimated_cost_usd: float = 0.0

    def add_usage(self, usage: Optional[Dict] = None, prompt: str = "", completion: str = "") -> None:
        """
        Add token usage from an OpenAI response or estimate from text length.

        Args:
            usage: OpenAI usage dict {'prompt_tokens': int, 'completion_tokens': int} (optional)
            prompt: Input prompt text (for fallback estimation)
            completion: Generated text (for fallback estimation)
        """
        if usage and "prompt_tokens" in usage and "completion_tokens" in usage:
            # Real usage from OpenAI response
            self.total_prompt_tokens += usage["prompt_tokens"]
            self.total_completion_tokens += usage["completion_tokens"]
        else:
            # Rough estimate: ~4 chars per token (very approximate)
            est_prompt = len(prompt) // 4 + 1
            est_completion = len(completion) // 4 + 1
            self.total_prompt_tokens += est_prompt
            self.total_completion_tokens += est_completion

        # Rough cost estimate (gpt-4o-mini pricing as of 2026)
        # $0.15 / 1M input tokens, $0.60 / 1M output tokens
        input_cost = self.total_prompt_tokens / 1_000_000 * 0.15
        output_cost = self.total_completion_tokens / 1_000_000 * 0.60
        self.estimated_cost_usd = input_cost + output_cost

    def get_summary(self) -> Dict:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }