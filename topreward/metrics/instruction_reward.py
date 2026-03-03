"""Instruction Reward Metric.

This module implements log-likelihood based instruction reward computation,
based on the approach from "Vision Language Models are In-Context Value Learners".

The instruction reward measures how well a trajectory of frames matches a given
instruction by computing the log-probability of generating the instruction text
conditioned on the video frames.
"""

from dataclasses import dataclass
from typing import Any

from topreward.metrics.base import MetricResult


@dataclass
class InstructionRewardResult:
    """Result from instruction reward computation.

    Attributes:
        reward: The computed reward (log-likelihood, possibly reduced).
        reduction: The reduction method used ("mean" or "sum").
        token_count: Number of instruction tokens scored.
        per_token_log_probs: Optional per-token log probabilities.
        token_ids: Optional token IDs corresponding to per_token_log_probs.
        trajectory_description: Optional trajectory description if use_video_description was True.
        prefix_lengths: Optional list of prefix lengths (frame counts) tested.
        prefix_rewards: Optional list of raw reward values for each prefix.
        normalized_prefix_rewards: Optional normalized (0-1) rewards for each prefix.
    """

    reward: float
    reduction: str
    token_count: int
    per_token_log_probs: list[float] | None = None
    token_ids: list[int] | None = None
    trajectory_description: str | None = None
    prefix_lengths: list[int] | None = None
    prefix_rewards: list[float] | None = None
    normalized_prefix_rewards: list[float] | None = None

    def to_metric_result(self) -> MetricResult:
        """Convert to standard MetricResult format."""
        return MetricResult(
            name="topreward",
            value=self.reward,
            details={
                "reduction": self.reduction,
                "token_count": self.token_count,
                "normalized_prefix_rewards": self.normalized_prefix_rewards,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "reward": self.reward,
            "reduction": self.reduction,
            "token_count": self.token_count,
        }
        if self.per_token_log_probs is not None:
            d["per_token_log_probs"] = self.per_token_log_probs
        if self.token_ids is not None:
            d["token_ids"] = self.token_ids
        if self.trajectory_description is not None:
            d["trajectory_description"] = self.trajectory_description
        if self.prefix_lengths is not None:
            d["prefix_lengths"] = self.prefix_lengths
        if self.prefix_rewards is not None:
            d["prefix_rewards"] = self.prefix_rewards
        if self.normalized_prefix_rewards is not None:
            d["normalized_prefix_rewards"] = self.normalized_prefix_rewards
        return d
