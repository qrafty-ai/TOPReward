"""HuggingFace/LeRobot data loader for TOPReward.

This module provides data loading from LeRobot datasets, supporting both:
- HuggingFace Hub datasets (requires authentication for private/gated datasets)
- Local datasets (load directly from disk without Hub access)

For local datasets:
- Provide the ``root`` parameter pointing to the local dataset directory
- Optionally provide ``dataset_name`` as repo_id (e.g. ``org/dataset``)
- If ``dataset_name`` is omitted, repo_id is inferred from local path

Example usage:
    # From HuggingFace Hub
    loader = HuggingFaceDataLoader(dataset_name="lerobot/aloha_static_coffee")

    # From local path (no Hub access needed)
    loader = HuggingFaceDataLoader(root="/path/to/local/dataset")
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
from datasets.utils.logging import disable_progress_bar
from loguru import logger
from numpy.typing import NDArray

from topreward.data_loaders.base import BaseDataLoader
from topreward.utils.data_types import Episode
from topreward.utils.data_types import Example as FewShotInput
from topreward.utils.video_utils import decode_video_frames

disable_progress_bar()


class HuggingFaceDataLoader(BaseDataLoader):
    """Load episodes from LeRobot datasets.

    Supports both HuggingFace Hub datasets and local datasets.
    For local datasets, provide ``root`` path. ``dataset_name`` is optional and,
    when provided, is used as the repo_id for LeRobot APIs.

    Produces a FewShotInput with one eval episode and up to ``num_context_episodes``
    sampled from the remaining pool. Frame count is controlled by ``num_frames``.
    """

    def __init__(
        self,
        *,
        dataset_name: str | None = None,
        root: str | None = None,
        camera_index: int = 0,
        num_frames: int = 20,
        num_context_episodes: int = 2,
        shuffle: bool = False,
        seed: int = 42,
        max_episodes: int | None = None,
        sampling_method: str = "random",
        anchoring: str = "first",
    ) -> None:
        super().__init__(
            num_frames=num_frames,
            num_context_episodes=num_context_episodes,
            shuffle=shuffle,
            seed=seed,
        )
        self.dataset_name = dataset_name
        self.root = os.path.expanduser(root) if root else None
        self.camera_index = int(camera_index)
        self.sampling_method = sampling_method
        self.anchoring = anchoring

        # Load dataset once (optimization #1: single dataset instance)
        if self.root:
            logger.info(f"Loading dataset from local path: {self.root}")
            self._load_local_dataset(dataset_name)
        else:
            if not dataset_name:
                raise ValueError("Either 'root' (local path) or 'dataset_name' (Hub repo) must be provided")
            logger.info(f"Loading dataset from HuggingFace Hub: {dataset_name}")
            self._load_hub_dataset(dataset_name)

        # Get total episodes
        self.max_episodes = min(max_episodes or self.ds_meta.total_episodes, self.ds_meta.total_episodes)

        # Pre-compute episode boundaries
        from lerobot.datasets.push_dataset_to_hub.utils import calculate_episode_data_index

        self._episode_data_index = calculate_episode_data_index(self._dataset.hf_dataset)
        logger.info(f"Pre-computed episode boundaries for {self.max_episodes} episodes")

        # Deterministic episode order
        self._all_episodes_indices = list(range(self.max_episodes))
        self._cursor = 0

    @staticmethod
    def _infer_repo_id_from_root(root: Path) -> str:
        if root.parent == root:
            raise ValueError(f"Could not infer repo_id from local root path: {root}")
        return f"{root.parent.name}/{root.name}"

    def _load_local_dataset(self, dataset_name: str | None) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

        if self.root is None:
            raise ValueError("Local loading requires 'root' to be set")

        root_path = Path(self.root)
        if not (root_path / "meta" / "info.json").exists():
            raise FileNotFoundError(
                f"LeRobot metadata not found at '{root_path / 'meta' / 'info.json'}'. Expected a local dataset root containing data/, meta/, and videos/."
            )

        repo_id = dataset_name or self._infer_repo_id_from_root(root_path)
        logger.info(f"Using local LeRobot dataset repo_id='{repo_id}' root='{root_path}'")

        self._dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root_path,
            force_cache_sync=False,
            download_videos=False,
        )
        self.ds_meta = LeRobotDatasetMetadata(
            repo_id=repo_id,
            root=root_path,
            force_cache_sync=False,
        )

    def _load_hub_dataset(self, dataset_name: str) -> None:
        """Load dataset from HuggingFace Hub."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

        self._dataset = LeRobotDataset(dataset_name, force_cache_sync=True)
        self.ds_meta = LeRobotDatasetMetadata(dataset_name)

    @property
    def fps(self) -> float:
        """Return the FPS from the LeRobot dataset metadata."""
        return float(self.ds_meta.fps)

    @property
    def total_episodes(self) -> int:
        return int(self.max_episodes)

    def _load_episode_frames(self, episode_index: int) -> tuple[list[NDArray[Any]], str]:
        """Load frames using batch video decoding for improved performance.

        Args:
            episode_index: Episode index to load frames from.

        Returns:
            Tuple of (frames_list, instruction_text)
        """
        # Get episode boundaries (from pre-computed index)
        from_idx = int(self._episode_data_index["from"][episode_index].item())
        to_idx = int(self._episode_data_index["to"][episode_index].item())

        logger.info(f"Loading episode [{episode_index}] frames from {from_idx} to {to_idx} (exclusive)")

        # Get camera key
        camera_key = self._dataset.meta.camera_keys[self.camera_index]

        # Batch-fetch timestamps from parquet (avoiding individual frame access)
        frame_indices = list(range(from_idx, to_idx))
        timestamps = self._dataset.hf_dataset["timestamp"]
        timestamps = [timestamps[idx].item() for idx in frame_indices]

        # Get video path
        video_path = self._dataset.root / self._dataset.meta.get_video_file_path(episode_index, camera_key)

        # BATCH DECODE all frames at once using optimized video codec
        # This is the key optimization - decode all frames in one operation
        frames_tensor = decode_video_frames(video_path, timestamps, self._dataset.tolerance_s, self._dataset.video_backend)

        # Convert to numpy HWC uint8 format expected by downstream code
        frames = []
        for frame in frames_tensor:
            frame_np = frame.numpy()
            # Convert CHW to HWC if needed
            if frame_np.shape[0] in [1, 3]:
                frame_np = np.transpose(frame_np, (1, 2, 0))
            # Convert to uint8 if needed
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            frames.append(frame_np)

        # Get instruction from task_index
        task_index = int(self._dataset.hf_dataset["task_index"][from_idx].item())
        instruction = str(self.ds_meta.tasks.index[task_index])

        return frames, instruction

    def _build_context(self, exclude_index: int) -> list[Episode]:
        pool = [i for i in self._all_episodes_indices if i != exclude_index]
        if not pool or self.num_context_episodes <= 0:
            return []
        # Deterministic sampling for the given eval episode
        rng = np.random.default_rng(self.seed + exclude_index)
        rng.shuffle(pool)
        chosen = pool[: self.num_context_episodes]
        ctx_eps: list[Episode] = []
        for idx in chosen:
            frames, instruction = self._load_episode_frames(idx)
            ctx_eps.append(self._build_episode(frames=frames, instruction=instruction, episode_index=idx))
        return ctx_eps

    def load_fewshot_input(self, episode_index: int | None = None) -> FewShotInput:
        if episode_index is None:
            if self._cursor >= len(self._all_episodes_indices):
                self._cursor = 0
            episode_index = self._all_episodes_indices[self._cursor]
            self._cursor += 1

        logger.info(f"Loading episode {episode_index} from {self.dataset_name or 'local dataset'}")
        frames, instruction = self._load_episode_frames(episode_index)
        eval_ep = self._build_episode(
            frames=frames, instruction=instruction, episode_index=episode_index, sampling_method=self.sampling_method, anchoring=self.anchoring
        )
        context = self._build_context(exclude_index=episode_index)
        return FewShotInput(eval_episode=eval_ep, context_episodes=context)

    def reset(self) -> None:
        self._cursor = 0
