"""Prediction script producing model inferences + metrics.

Steps:
1. Instantiate data loader & model client via Hydra.
2. Sample N examples (FewShotInput) from loader.
3. For each example, call the shared prediction helper.
4. Persist JSONL outputs (one line per example) + aggregated metrics summary.

Supports two methods:
- gvl: Generative Value Learning (predicts task completion percentages)
- topreward: Log-likelihood reward for instruction matching
"""

import json
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from topreward.clients.base import BaseModelClient
from topreward.data_loaders.base import BaseDataLoader
from topreward.mapper.base import BaseMapper
from topreward.metrics.voc import VOCMetric
from topreward.results.prediction import aggregate_metrics, summarize_failures
from topreward.utils import inference as infer_utils
from topreward.utils.logging_config import setup_logging


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="experiments/predict_gvl",
)
def main(config: DictConfig) -> None:
    """Main prediction script entry point."""
    # Configure logging format
    setup_logging(level="INFO", format_type="detailed")

    infer_utils.validate_prediction_config(config)
    load_dotenv(override=True)
    logger.info("Environment variables loaded (dotenv)")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    data_loader: BaseDataLoader = instantiate(config.data_loader)
    client: BaseModelClient = instantiate(config.model)
    prompt_template: str = config.prompts.template

    # Get prediction method (gvl or topreward). Keep
    # `instruction_reward` as a backward-compatible alias.
    method = str(config.prediction.get("method", "gvl")).lower()

    if method not in ("gvl", "topreward"):
        raise ValueError(f"Unknown prediction method: {method}. Use 'gvl' or 'topreward'.")

    # Only instantiate mapper for gvl method (topreward doesn't
    # need it)
    mapper: BaseMapper | None = None
    if method == "gvl":
        mapper = instantiate(config.mapper)
        logger.info(
            f"Instantiated components | dataset={config.dataset.name} "
            f"loader={data_loader.__class__.__name__} "
            f"model={client.__class__.__name__} "
            f"mapper={mapper.__class__.__name__} method={method} "
            f"prompt_template_chars={len(prompt_template)}"
        )
    else:
        logger.info(
            f"Instantiated components | dataset={config.dataset.name} loader={data_loader.__class__.__name__} model={client.__class__.__name__} method={method}"
        )

    num_examples = int(config.prediction.num_examples)
    eval_all_episodes = bool(config.prediction.get("eval_all_episodes", False))
    if eval_all_episodes:
        total_episodes = data_loader.total_episodes
        if total_episodes is None:
            logger.warning("eval_all_episodes requested but data loader does not expose total_episodes; using num_examples.")
        else:
            num_examples = int(total_episodes)
            logger.info(f"eval_all_episodes enabled; overriding num_examples to {num_examples}")
    save_raw = bool(config.prediction.save_raw)
    output_dir = Path(str(config.prediction.output_dir))
    logger.info(f"Predictions will be saved to directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name_safe = client.model_name.replace("/", "_")
    starting_time = datetime.now().isoformat().replace(":", "-")
    resume_from_path = config.prediction.get("resume_from_path")
    resume_from_index = config.prediction.get("resume_from_index")
    if resume_from_index is not None:
        resume_from_index = int(resume_from_index)

    if resume_from_path:
        output_path = Path(str(resume_from_path))
        if not output_path.is_absolute():
            output_path = output_dir / output_path
    else:
        output_path = output_dir / f"{model_name_safe}_{starting_time}_predictions.jsonl"
    sampling_method = config.sampling_method
    anchoring = config.anchoring

    voc_metric = VOCMetric()
    logger.debug(f"Metrics initialized: {voc_metric.name}")

    # Load prompt phrasing from dedicated config section (required; fall
    # back to empty)
    prompt_phrases = dict(config.get("prompt_phrases", {})) if hasattr(config, "prompt_phrases") else {}
    logger.debug(f"Prompt phrases: {prompt_phrases}")

    if resume_from_path and output_path.exists():
        inferred_resume_index = 0
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                idx = payload.get("index")
                if isinstance(idx, int):
                    inferred_resume_index = max(inferred_resume_index, idx + 1)
        if resume_from_index is None:
            resume_from_index = inferred_resume_index
        elif resume_from_index < inferred_resume_index:
            logger.warning(f"Resume index {resume_from_index} is behind existing file max index {inferred_resume_index - 1}; new records may overlap.")

    if resume_from_index is None:
        resume_from_index = 0

    remaining_examples = max(num_examples - resume_from_index, 0)
    logger.info(
        f"Processing {remaining_examples}/{num_examples} examples using method='{method}' (resume_from_index={resume_from_index}, output={output_path})"
    )

    records = []
    # Create/truncate the output file up front when not resuming, then append
    # per-example to avoid holding an open buffer across the full run.
    if not (resume_from_path and output_path.exists()):
        with output_path.open("w", encoding="utf-8"):
            pass

    # TOPReward-specific config
    ir_reduction = str(config.prediction.get("reduction", "mean"))
    ir_use_video_description = bool(config.prediction.get("use_video_description", False))
    ir_use_subsampled_video = bool(config.prediction.get("use_subsampled_video", False))
    ir_add_chat_template = bool(config.prediction.get("add_chat_template", False))
    # Get FPS from dataset, fall back to config override if specified
    ir_fps = data_loader.fps

    if remaining_examples <= 0:
        logger.warning("No remaining examples to process after applying resume_from_index.")

    if resume_from_index > 0 and remaining_examples > 0:
        logger.info(f"Advancing data loader by {resume_from_index} examples to resume.")
        for _ in tqdm(range(resume_from_index), desc="Skipping"):
            data_loader.load_fewshot_input()

    for offset in tqdm(range(remaining_examples), desc=f"Predicting ({method})"):
        idx = resume_from_index + offset
        if method == "gvl":
            ex = data_loader.load_fewshot_input()
            if mapper is None:
                raise ValueError("Mapper must be instantiated for gvl method")
            record = infer_utils.predict_on_fewshot_input(
                idx,
                num_examples,
                ex,
                client,
                prompt_template,
                save_raw,
                voc_metric,
                config.dataset.name,
                temperature=float(config.prediction.get("temperature", 0.0)),
                mapper=mapper,
                prompt_phrases=prompt_phrases,
            )
            with output_path.open("a", encoding="utf-8") as output_file:
                output_file.write(json.dumps(record.to_dict(include_images=False), ensure_ascii=False) + "\n")

            # Clear image data to save memory - keep only metadata needed
            # for aggregation
            record.example.eval_episode.shuffled_frames = []
            record.example.eval_episode.starting_frame = None
            record.example.eval_episode.all_frames = None
            for ctx_ep in record.example.context_episodes:
                ctx_ep.shuffled_frames = []
                ctx_ep.starting_frame = None
                ctx_ep.all_frames = None
            record.raw_response = None  # Also clear raw response if saved

        else:  # topreward
            ex = data_loader.load_fewshot_input()
            record = infer_utils.compute_instruction_reward_on_fewshot_input(
                idx,
                num_examples,
                ex,
                client,
                config.dataset.name,
                reduction=ir_reduction,
                fps=ir_fps,
                use_video_description=ir_use_video_description,
                use_subsampled_video=ir_use_subsampled_video,
                add_chat_template=ir_add_chat_template,
            )
            with output_path.open("a", encoding="utf-8") as output_file:
                output_file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

        records.append(record)

    logger.info(f"Wrote {len(records)} records to {output_path}")

    # Build summary based on method
    summary = {}
    summary["model_name"] = client.model_name
    summary["dataset_name"] = config.dataset.name
    summary["num_context_episodes"] = config.dataset.num_context_episodes
    summary["prediction_time"] = starting_time
    summary["method"] = method
    summary["num_examples"] = len(records)
    summary["num_examples_requested"] = num_examples
    summary["eval_all_episodes"] = eval_all_episodes
    summary["resume_from_index"] = resume_from_index
    if resume_from_path:
        summary["resume_from_path"] = str(output_path)
    summary["sampling"] = sampling_method
    summary["anchoring"] = anchoring

    if method == "gvl":
        failure_count, failure_totals = summarize_failures(records)
        dataset_metrics = aggregate_metrics(records)
        logger.success(
            f"Aggregate metrics (excluding failures): "
            f"used={dataset_metrics.total_examples} "
            f"valid={dataset_metrics.valid_predictions} "
            f"ratio={dataset_metrics.length_valid_ratio if dataset_metrics.length_valid_ratio is not None else 0.0:.2f} "
            f"voc_mean={dataset_metrics.metric_means.get('voc', float('nan')):.4f}"
        )
        logger.info(f"Failures (excluded from aggregates): {failure_count}/{len(records)} | breakdown={failure_totals}")
        summary["temperature"] = float(config.prediction.get("temperature", 1.0))
        summary["failure_count"] = failure_count
        summary["failure_breakdown"] = failure_totals
        summary["num_examples_used_for_metrics"] = dataset_metrics.total_examples
        summary["metrics"] = dataset_metrics.to_dict()
        summary["prompt_type"] = config.prompts.name
    else:  # topreward
        # Compute aggregate stats for TOPReward
        valid_records = [r for r in records if r.error is None]
        error_records = [r for r in records if r.error is not None]
        rewards = [r.reward for r in valid_records]
        normalized_rewards = [r.normalized_log_probs for r in valid_records if r.normalized_log_probs is not None]

        if rewards:
            mean_reward = float(np.mean(rewards))
            std_reward = float(np.std(rewards))
            min_reward = float(np.min(rewards))
            max_reward = float(np.max(rewards))
        else:
            mean_reward = std_reward = min_reward = max_reward = float("nan")

        if normalized_rewards:
            # Flatten list of lists to compute global min/max
            all_normalized = [v for norms in normalized_rewards for v in norms]
            min_normalized = float(np.min(all_normalized))
            max_normalized = float(np.max(all_normalized))
        else:
            min_normalized = max_normalized = float("nan")

        # Collect per-record VOC scores (already computed in TOPReward path)
        voc_scores = [r.voc for r in valid_records if r.voc is not None]
        if voc_scores:
            mean_voc = float(np.mean(voc_scores))
            std_voc = float(np.std(voc_scores))
            min_voc = float(np.min(voc_scores))
            max_voc = float(np.max(voc_scores))
        else:
            mean_voc = std_voc = min_voc = max_voc = float("nan")

        logger.success(
            f"TOPReward metrics: "
            f"valid={len(valid_records)}/{len(records)} "
            f"mean={mean_reward:.4f} std={std_reward:.4f} "
            f"range=[{min_reward:.4f}, {max_reward:.4f}]"
        )
        logger.success(f"Normalized reward range=[{min_normalized:.4f}, {max_normalized:.4f}]")
        logger.success(f"VOC metrics: n={len(voc_scores)} mean={mean_voc:.4f} std={std_voc:.4f} range=[{min_voc:.4f}, {max_voc:.4f}]")
        if error_records:
            logger.warning(f"Errors: {len(error_records)}/{len(records)} records failed")

        summary["reduction"] = ir_reduction
        summary["fps"] = ir_fps
        summary["use_video_description"] = ir_use_video_description
        summary["use_subsampled_video"] = ir_use_subsampled_video
        summary["add_chat_template"] = ir_add_chat_template
        summary["valid_count"] = len(valid_records)
        summary["error_count"] = len(error_records)
        summary["metrics"] = {
            "topreward_mean": mean_reward,
            "topreward_std": std_reward,
            "topreward_min": min_reward,
            "topreward_max": max_reward,
            "voc_mean": mean_voc,
            "voc_std": std_voc,
            "voc_min": min_voc,
            "voc_max": max_voc,
            "voc_valid_count": len(voc_scores),
        }

    with (output_dir / f"{model_name_safe}_{starting_time}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote {len(records)} records to {output_path}")
    logger.info(f"Summary: {summary}")


if __name__ == "__main__":  # pragma: no cover
    main()
