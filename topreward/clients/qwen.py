from collections.abc import Sequence
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from topreward.clients.base import BaseModelClient
from topreward.metrics.instruction_reward import InstructionRewardResult
from topreward.utils.aliases import Event, ImageEvent, ImageNumpy, ImageT, TextEvent
from topreward.utils.constants import MAX_TOKENS_TO_GENERATE
from topreward.utils.images import to_pil


class QwenClient(BaseModelClient):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        rpm: float = 0.0,
        max_input_length: int = 32768,
    ):
        super().__init__(rpm=rpm)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        logger.info(type(self.processor))
        self.model_name = model_name
        self.max_input_length = max_input_length

    def _generate_from_events(self, events: list[Event], temperature: float) -> str:
        messages = [{"role": "user", "content": []}]
        for ev in events:
            if isinstance(ev, TextEvent):
                messages[0]["content"].append({"type": "text", "text": ev.text})
            elif isinstance(ev, ImageEvent):
                messages[0]["content"].append({"type": "image", "image": to_pil(cast(ImageT, ev.image))})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)  # type: ignore[misc]

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        input_len = inputs["input_ids"].shape[-1]
        if input_len > self.max_input_length:
            raise ValueError(f"Input length {input_len} exceeds maximum of {self.max_input_length} tokens")
        logger.info(f"Input length: {input_len}")

        # Inference: Generation of the output
        if temperature == 0.0:
            generated_ids = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS_TO_GENERATE, do_sample=False)
        else:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS_TO_GENERATE,
                do_sample=True,
                temperature=temperature,
            )
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]

    def generate_object_state_reasoning(
        self,
        frames: Sequence[ImageT],
        fps: float = 2.0,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate a description of the robot manipulation trajectory.

        This generates an instruction-agnostic description to avoid circular dependencies
        where mentioning instruction objects would artificially inflate likelihood scores.

        Args:
            frames: List of images representing the video.
            fps: Frames per second for video input (default: 2.0).
            max_new_tokens: Maximum tokens to generate for reasoning.

        Returns:
            Generated text describing the trajectory.
        """
        # Convert frames to PIL images
        pil_frames = [to_pil(cast(ImageT, f)) for f in frames]

        content = [
            {"type": "video", "video": pil_frames, "fps": fps},
            {
                "type": "text",
                "text": "Describe the robot manipulation trajectory in this video:",
            },
        ]

        user_messages = [{"role": "user", "content": content}]

        # Apply chat template
        prompt_chat = self.processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)

        # Process vision info
        image_inputs, video_inputs = process_vision_info(user_messages)  # type: ignore[misc]

        # Prepare inputs
        inputs = self.processor(
            text=[prompt_chat],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        # Generate description
        self.model.eval()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )

        # Decode response
        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Extract only the generated part (after the prompt)
        # The response includes the prompt, so we need to extract just the
        # generated text

        prompt_text = self.processor.batch_decode(
            inputs["input_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        description = response[len(prompt_text) :].strip() if response.startswith(prompt_text) else response.strip()

        return description

    def compute_instruction_reward(
        self,
        frames: list[ImageNumpy],
        instruction: str,
        reduction: str = "mean",
        fps: float = 2.0,
        use_video_description: bool = False,
        add_chat_template: bool = False,
    ) -> InstructionRewardResult:
        """Compute a log-likelihood reward for an instruction conditioned on a trajectory of frames.

        This implements the instruction reward approach from "Vision Language Models are
        In-Context Value Learners", measuring how well the trajectory matches the given
        instruction by computing the log-probability of generating the instruction text.

        Args:
            frames: List of images representing the trajectory (at least 2 frames).
            instruction: Instruction text to evaluate.
            reduction: Reduction to apply to token log probabilities ("mean" or "sum").
            fps: Frames per second for video input (default: 2.0).
            use_video_description: If True, generate instruction-agnostic description of
                                  the robot manipulation trajectory, then prepend it as context
                                  before evaluating instruction likelihood. This avoids circular
                                  dependencies that would artificially inflate scores.
            add_chat_template: If True, wrap the full prompt (including instruction) with
                               the chat template before tokenization.

        Returns:
            InstructionRewardResult with the computed reward and metadata.
        """

        pil_frames = [to_pil(cast(ImageT, f)) for f in frames]

        # Optionally generate trajectory description for augmented context
        trajectory_description = None
        if use_video_description:
            logger.info("Generating trajectory description...")
            trajectory_description = self.generate_object_state_reasoning(frames, fps=fps)
            logger.info("Generated trajectory description.")
            logger.info("Trajectory description: %s", trajectory_description)
            # Prepend trajectory description as object state reasoning
            prompt_text = (
                f"{trajectory_description} Therefore given the above "
                f"description and the video, the video shows a robot "
                f"manipulation trajectory that **completes** the following "
                f"instruction: "
            )
        else:
            # Original prompt without description
            prompt_text = "The above video shows a robot manipulation trajectory that completes the following task: "

        content = [
            {"type": "video", "video": pil_frames, "fps": fps},
            {"type": "text", "text": prompt_text},
        ]
        user_messages = [{"role": "user", "content": content}]
        eos_token = self.processor.tokenizer.eos_token

        if add_chat_template:
            instruction_suffix = f"{instruction} Decide whether the above statement is True or not. The answer is:"
            templated_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": pil_frames, "fps": fps},
                        {"type": "text", "text": f"{prompt_text}{instruction_suffix}"},
                    ],
                }
            ]
            prompt_chat = self.processor.apply_chat_template(
                templated_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = f"{prompt_chat}True"
            image_inputs, video_inputs = process_vision_info(templated_messages)  # type: ignore[misc]
        else:
            instruction_suffix = f"{instruction} Decide whether the above statement is True or not. The answer is: True"
            prompt_chat = self.processor.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if eos_token is not None:
                prompt_chat = prompt_chat.split(eos_token)[0]
            full_text = f"{prompt_chat}{instruction_suffix}"
            image_inputs, video_inputs = process_vision_info(user_messages)  # type: ignore[misc]

        inputs = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")
        labels = inputs["input_ids"].clone()

        # Mask the prompt so we only compute loss on the instruction + "True" part
        prompt_length = inputs["input_ids"].shape[1] - 1
        labels[:, :prompt_length] = -100
        if "attention_mask" in inputs:
            labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)

        # Compute per-token log probabilities
        logits = outputs.logits[:, :-1, :]
        target_labels = labels[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        mask = target_labels != -100
        safe_targets = target_labels.masked_fill(~mask, 0)
        token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        masked_log_probs = token_log_probs[mask]

        # Apply reduction
        reward = masked_log_probs.sum().item() if reduction == "sum" else masked_log_probs.mean().item()

        # Extract per-token info for metadata
        per_token_log_probs_list = masked_log_probs.detach().cpu().tolist()
        token_ids_list = target_labels[mask].detach().cpu().tolist()

        return InstructionRewardResult(
            reward=reward,
            reduction=reduction,
            token_count=len(per_token_log_probs_list),
            per_token_log_probs=per_token_log_probs_list,
            token_ids=token_ids_list,
            trajectory_description=trajectory_description,
        )

    @staticmethod
    def normalize_rewards(
        rewards: Sequence[float],
        method: str = "minmax",
    ) -> np.ndarray:
        """Normalize a sequence of instruction rewards to a 0-1 range.

        This is useful for comparing rewards across different trajectories or
        trajectory prefixes, as raw log-likelihood values can be hard to interpret.

        Args:
            rewards: Sequence of raw reward values (log-likelihoods).
            method: Normalization method. Options:
                - "minmax": Scale to [0, 1] using min-max normalization.

        Returns:
            Normalized rewards as a numpy array in [0, 1] range.
        """
        rewards_arr = np.array(rewards, dtype=np.float64)

        if len(rewards_arr) == 0:
            return rewards_arr

        if len(rewards_arr) == 1:
            return np.array([1.0])

        if method == "minmax":
            r_min, r_max = rewards_arr.min(), rewards_arr.max()
            if r_max == r_min:
                # All rewards are identical
                return np.ones_like(rewards_arr)
            return (rewards_arr - r_min) / (r_max - r_min)

        else:
            raise ValueError(f"Unknown normalization method: {method}. Use 'minmax' or 'softmax'.")

    def compute_instruction_rewards_for_prefixes(  # type: ignore[override]
        self,
        frames: list[ImageNumpy],
        instruction: str,
        num_samples: int = 15,
        reduction: str = "mean",
        fps: float = 2.0,
        use_video_description: bool = False,
        add_chat_template: bool = False,
    ) -> InstructionRewardResult:
        """Compute instruction rewards for trajectory prefixes of varying lengths.

        This is useful for analyzing how reward changes as the trajectory progresses,
        similar to the analysis in the demo script.

        Args:
            frames: Full list of trajectory frames.
            instruction: Instruction text to evaluate.
            num_samples: Number of prefix lengths to sample (uniformly spaced).
            reduction: Reduction method ("mean" or "sum").
            fps: Frames per second for video input.
            add_chat_template: Whether to wrap the instruction prompt with the chat template.

        Returns:
            InstructionRewardResult with prefix_lengths, prefix_rewards, and normalized_prefix_rewards.
            The main reward is the full trajectory reward (last prefix).
        """

        num_frames = len(frames)
        num_samples = min(num_samples, num_frames)

        # Generate uniformly spaced prefix lengths from 2 to full trajectory
        if num_frames > 2:
            prefix_lengths = np.linspace(1, num_frames, num_samples, dtype=int)
            prefix_lengths = sorted({int(x) for x in prefix_lengths})
        else:
            prefix_lengths = [num_frames]

        rewards = []
        token_counts = []

        for length in prefix_lengths:
            prefix_frames = frames[:length]
            result = self.compute_instruction_reward(
                frames=prefix_frames,
                instruction=instruction,
                reduction=reduction,
                fps=fps,
                use_video_description=use_video_description,
                add_chat_template=add_chat_template,
            )
            rewards.append(result.reward)
            token_counts.append(result.token_count)
            logger.info(f"Prefix {length:3d} frames: reward={result.reward:.4f} ({result.token_count} tokens)")

        normalized_rewards = self.normalize_rewards(rewards).tolist()

        # Full trajectory is the last prefix
        return InstructionRewardResult(
            reward=rewards[-1],
            reduction=reduction,
            token_count=token_counts[-1],
            prefix_lengths=list(prefix_lengths),
            prefix_rewards=rewards,
            normalized_prefix_rewards=normalized_rewards,
        )
