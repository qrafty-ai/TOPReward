import inspect
import json
from collections.abc import Iterable
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig

from topreward.clients.base import BaseModelClient
from topreward.mapper.base import BaseMapper
from topreward.metrics.base import MetricResult
from topreward.metrics.voc import VOCMetric, value_order_correlation
from topreward.results.prediction import InstructionRewardRecord, PredictionRecord
from topreward.utils.aliases import ImageNumpy
from topreward.utils.constants import N_DEBUG_PROMPT_CHARS
from topreward.utils.data_types import Example as FewShotInput
from topreward.utils.data_types import InferredEpisode, InferredFewShotResult
from topreward.utils.errors import MaxRetriesExceededError, PercentagesCountMismatchError, PercentagesNormalizationError
from topreward.utils.hydra import ensure_required_keys
from topreward.utils.prompts import format_prompt


def build_inferred_example(
    fewshot: FewShotInput,
    predicted: list[int],
) -> InferredFewShotResult:
    inferred_ep = InferredEpisode.from_predictions(fewshot.eval_episode, predictions=predicted)
    return InferredFewShotResult(eval_episode=inferred_ep, context_episodes=fewshot.context_episodes)


def save_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def validate_prediction_config(config: DictConfig) -> None:
    """Ensure required top-level keys are present for prediction runs.

    This mirrors the previous local _validate_config in the script.
    """
    for key in ("dataset", "data_loader", "model", "prompts", "prediction"):
        ensure_required_keys(config, key)


def load_fewshot_examples(loader, n: int, dataset_name: str) -> list[FewShotInput]:
    """Load N few-shot inputs from a data loader with logging.

    Args:
        loader: Instance of BaseDataLoader.
        n: Number of examples to load.
        dataset_name: Human-friendly dataset identifier for logs.
    Returns:
        List of FewShotInput objects.
    """
    logger.info(f"Generating {n} examples…")
    examples: list[FewShotInput] = []
    for i in range(n):
        logger.info(f"Loading example {i + 1}/{n}")
        ex = loader.load_fewshot_input()
        examples.append(ex)
    logger.success(f"Loaded {len(examples)} few-shot examples from dataset '{dataset_name}'")
    return examples


def predict_on_fewshot_input(
    idx: int,
    total: int,
    ex: FewShotInput,
    client: BaseModelClient,
    prompt_template: str,
    save_raw: bool,
    voc_metric: VOCMetric,
    dataset_name: str,
    temperature: float,
    mapper: BaseMapper,
    *,
    prompt_phrases: dict[str, str] | None = None,
) -> PredictionRecord:
    """Run model prediction and metric computation on a single few-shot input.

    The logic mirrors the original script function without changes.
    """
    logger.info(f"Processing example {idx + 1}/{total} (episode_index={ex.eval_episode.episode_index}) from {dataset_name}")
    prompt = format_prompt(prompt_template, instruction=ex.eval_episode.instruction)
    logger.debug(f"Prompt (truncated {N_DEBUG_PROMPT_CHARS} chars): {prompt[:N_DEBUG_PROMPT_CHARS]}...")

    logger.info(f"PROMPT for example {idx}:\n{prompt}")

    try:
        response_text = client.generate_response(
            prompt,
            ex.eval_episode,
            ex.context_episodes,
            temperature=temperature,
            prompt_phrases=(prompt_phrases or {}),
        )
    except (RuntimeError, ValueError, OSError, MaxRetriesExceededError) as e:
        logger.error(f"Model generation failed for example {idx}: {e}")
        predicted: list[float] = []
        response_text = f"<error: {e}>"
    logger.debug(f"Response on example {idx}:\n{response_text}")

    expected_len = len(ex.eval_episode.shuffled_frames)
    error_count: dict[str, int] = {
        PercentagesCountMismatchError.__name__: 0,
        PercentagesNormalizationError.__name__: 0,
    }

    try:
        predicted = mapper.extract_percentages(response_text)
        logger.success(f"Extracted {len(predicted)} percentages on example {idx}")
    except PercentagesNormalizationError as e:
        logger.error(f"Extraction error on example {idx}: {e}")
        predicted = []
        error_count[PercentagesNormalizationError.__name__] += 1

    if len(predicted) != expected_len:
        logger.error(f"Count mismatch on example {idx}: expected {expected_len}, got {len(predicted)}")
        error_count[PercentagesCountMismatchError.__name__] += 1

    inferred: InferredFewShotResult = build_inferred_example(ex, [int(p) for p in predicted])

    if sum(error_count.values()) > 0:
        metric_res = MetricResult(
            name=voc_metric.name,
            value=0,
            details={"note": (f"errors in prediction prevented metric computation {error_count!s}")},
        )
    else:
        metric_res = voc_metric.compute(inferred)
    metrics_payload = {metric_res.name: metric_res.value}

    if metric_res.details:
        for k, v in metric_res.details.items():
            metrics_payload[f"{metric_res.name}_{k}"] = v

    logger.debug(
        f"Metrics example {idx}: {metric_res.name}="
        f"{(metric_res.value if metric_res.value is not None else float('nan')):.4f}"
        f"{(' details=' + str(metric_res.details)) if metric_res.details else ''}"
    )

    record = PredictionRecord(
        index=idx,
        dataset=dataset_name,
        example=inferred,
        predicted_percentages=[float(p) for p in predicted],
        valid_length=len(predicted) == len(ex.eval_episode.shuffled_frames),
        metrics=metrics_payload,
        raw_response=response_text if save_raw else None,
        error_count=error_count,
    )
    logger.info(f"Example {idx}: preds={len(predicted)}/{len(ex.eval_episode.shuffled_frames)} VOC={metric_res.value}")
    return record


def compute_instruction_reward_on_fewshot_input(
    idx: int,
    total: int,
    ex: FewShotInput,
    client: BaseModelClient,
    dataset_name: str,
    reduction: str = "mean",
    fps: float | None = None,
    use_video_description: bool = False,
    use_subsampled_video: bool = False,
    use_video_input: bool = True,
    add_chat_template: bool = False,
) -> InstructionRewardRecord:
    """Compute instruction reward for a single few-shot input.

    This uses the client's compute_instruction_reward method to evaluate how well
    the trajectory matches the instruction by computing log-likelihood.

    Args:
        idx: Index of the example.
        total: Total number of examples.
        ex: FewShotInput containing the episode to evaluate.
        client: Model client (must support compute_instruction_reward).
        dataset_name: Name of the dataset for logging.
        reduction: Reduction method ("mean" or "sum").
        fps: Frames per second for video input. If None, computed automatically
            based on number of frames (assumes ~10 second trajectory).
        use_video_description: If True, generate trajectory description before
            computing reward to provide additional context.
        use_subsampled_video: If True, use subsampled frames. If False, use all frames.
        use_video_input: If True (Gemini only), send frames as video. If False, send as images.
        add_chat_template: If True and supported by the client, wrap the prompt with
            the model's chat template when computing instruction rewards.

    Returns:
        InstructionRewardRecord with the computed reward.
    """
    from topreward.results.prediction import InstructionRewardRecord

    logger.info(f"Computing instruction reward {idx + 1}/{total} (episode_index={ex.eval_episode.episode_index}) from {dataset_name}")

    frames: list[ImageNumpy]
    if use_subsampled_video:
        # Get frames in CHRONOLOGICAL order, interpolating uniformly spaced
        # frames. Temporal order is critical for understanding the trajectory
        # Include starting frame + all selected frames in original order
        uniformly_spaced_frames = ex.eval_episode.get_uniformly_spaced_frames()
        if ex.eval_episode.starting_frame is None:
            raise ValueError("starting_frame must not be None for instruction reward computation")
        frames = [ex.eval_episode.starting_frame, *uniformly_spaced_frames]
        instruction = ex.eval_episode.instruction
    else:
        # Use all frames as-is (starting frame + shuffled frames)
        all_frames = ex.eval_episode.all_frames
        if all_frames is None:
            raise ValueError(f"Episode {ex.eval_episode.episode_index} has no frames (all_frames is None)")
        frames = all_frames
        instruction = ex.eval_episode.instruction

    if fps is None:
        raise ValueError(f"fps must be provided for instruction reward computation on example {idx}")
    error_msg = None
    result = None

    # Compute rewards for trajectory prefixes (first N frames)
    # This gives us a list of rewards showing how reward changes over
    # trajectory

    # Build kwargs - only pass use_video_input if client supports it (Gemini only)
    kwargs = {
        "frames": frames,
        "instruction": instruction,
        "num_samples": 15,
        "reduction": reduction,
        "fps": fps,
        "use_video_description": use_video_description,
    }

    # Check if client supports use_video_input parameter (Gemini-specific)
    sig = inspect.signature(client.compute_instruction_rewards_for_prefixes)
    if "use_video_input" in sig.parameters:
        kwargs["use_video_input"] = use_video_input
    if "add_chat_template" in sig.parameters:
        kwargs["add_chat_template"] = add_chat_template

    logger.info(f"Example {idx}/{total}: Computing instruction rewards for {len(frames)} frames with {kwargs['num_samples']} prefix samples...")
    result = client.compute_instruction_rewards_for_prefixes(**kwargs)

    # Log prefix rewards
    logger.info(f"Reward by Trajectory Prefix (first N of {len(frames)} frames):")
    for length, r, norm_r in zip(
        result.prefix_lengths or [],
        result.prefix_rewards or [],
        result.normalized_prefix_rewards or [],
        strict=False,
    ):
        logger.info(f"  First {length:3d} frames: reward = {r:8.4f} (normalized: {norm_r:.4f})")

    logger.success(f"Example {idx}: topreward={result.reward:.4f} (tokens={result.token_count}, reduction={reduction}, fps={fps:.2f})")

    # Compute VOC: Spearman correlation between normalized rewards
    # and ground truth completion rates aligned to prefix lengths.
    voc_score = None
    if result is not None and result.normalized_prefix_rewards is not None:
        true_progress = ex.eval_episode.original_frames_task_completion_rates
        normalized_rewards = result.normalized_prefix_rewards
        voc_score = value_order_correlation(normalized_rewards, true_progress)
        voc_score = float(voc_score)
        logger.info(f"Example {idx}: VOC={voc_score:.4f}")
    else:
        logger.info(f"Example {idx}: Skipping VOC computation (no normalized rewards)")

    record = InstructionRewardRecord(
        index=idx,
        dataset=dataset_name,
        episode_index=ex.eval_episode.episode_index,
        instruction=instruction,
        reward=result.reward if result else float("nan"),
        reduction=reduction,
        token_count=result.token_count if result else 0,
        num_frames=len(frames),
        error=error_msg,
        normalized_log_probs=(result.normalized_prefix_rewards if result else None),
        voc=voc_score,
        original_frames_indices=ex.eval_episode.original_frames_indices,
        original_frames_task_completion_rates=(ex.eval_episode.original_frames_task_completion_rates),
        trajectory_description=(result.trajectory_description if result else None),
        prefix_lengths=result.prefix_lengths if result else None,
        prefix_rewards=result.prefix_rewards if result else None,
    )
    logger.info(f"Example {idx}: Record created, moving to next example")
    return record
