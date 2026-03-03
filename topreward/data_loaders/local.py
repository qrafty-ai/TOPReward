from collections.abc import Sequence
from pathlib import Path

import cv2
from loguru import logger
from PIL import Image

from topreward.data_loaders.base import BaseDataLoader
from topreward.utils.data_types import Example as FewShotInput

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


class LocalDataLoader(BaseDataLoader):
    """Load local episodes from image files or video files.

    Supported inputs:
    - `episodes_files`: explicit frame paths per episode (list of list of image paths)
    - `video_path`: one local video file for a single episode
    - `episode_videos`: one video path per episode
    """

    def __init__(
        self,
        *,
        episodes_files: Sequence[Sequence[str]] | None = None,
        video_path: str | None = None,
        episode_videos: Sequence[str] | None = None,
        instruction: str = "",
        num_frames: int = 20,
        num_context_episodes: int = 0,
        shuffle: bool = False,
        seed: int = 42,
        sampling_method: str = "random",
    ) -> None:
        super().__init__(
            num_frames=num_frames,
            num_context_episodes=num_context_episodes,
            shuffle=shuffle,
            seed=seed,
        )
        self.episodes_files: list[list[str]] = [list(ep) for ep in episodes_files] if episodes_files else []
        self.episode_videos: list[str] = list(episode_videos) if episode_videos else []
        if video_path:
            self.episode_videos.insert(0, video_path)

        if not self.episodes_files and not self.episode_videos:
            raise ValueError("Provide at least one of: episodes_files, video_path, episode_videos")

        self.instruction = instruction or ""
        self.sampling_method = sampling_method
        self._fps = self._probe_first_video_fps()

    def _load_images(self, paths: list[Path]):
        images = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning(f"Skipping unreadable image {p}: {exc}")
        return images

    def _load_video_frames(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open local video: {video_path}")

        frames = []
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                # OpenCV decodes BGR; convert to RGB for consistency.
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        finally:
            cap.release()

        return frames

    def _probe_first_video_fps(self) -> float:
        if not self.episode_videos:
            return 1.0
        cap = cv2.VideoCapture(str(self.episode_videos[0]))
        if not cap.isOpened():
            return 1.0
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            return fps if fps > 0 else 1.0
        finally:
            cap.release()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_episodes(self) -> int:
        return len(self.episode_videos) if self.episode_videos else len(self.episodes_files)

    def load_fewshot_input(self, episode_index: int | None = None) -> FewShotInput:
        if episode_index is None:
            episode_index = 0
        if episode_index < 0 or episode_index >= self.total_episodes:
            raise IndexError(f"episode_index {episode_index} out of range [0, {self.total_episodes})")

        if self.episode_videos:
            video_path = Path(self.episode_videos[episode_index])
            frames = self._load_video_frames(video_path)
        else:
            # Do not reorder or auto-discover; respect user-provided order strictly.
            paths = [Path(p) for p in self.episodes_files[episode_index]]
            frames = self._load_images(paths)

        if not frames:
            raise ValueError(f"No readable frames found for episode {episode_index}")
        ep = self._build_episode(frames=frames, instruction=self.instruction, episode_index=episode_index or 0, sampling_method=self.sampling_method)
        return FewShotInput(eval_episode=ep, context_episodes=[])
