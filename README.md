# TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics

[![Paper](https://img.shields.io/badge/arXiv-2602.19313-b31b1b.svg)](https://arxiv.org/pdf/2602.19313)
[![Website](https://img.shields.io/badge/Website-topreward.github.io-green)](https://topreward.github.io/webpage/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<!-- cspell:disable-next-line -->
**Shirui Chen, Cole Harrison, Ying-Chun Lee, Angela Jin Yang, Zhongzheng Ren, Lillian J. Ratliff, Jiafei Duan\*, Dieter Fox\*, Ranjay Krishna\***

<sup>*Equal advising</sup>

TOPReward is a reward modeling method that uses token probabilities from vision-language models (VLMs) as zero-shot reward signals for robotics. By computing the log-likelihood that a model assigns to a task instruction given a video trajectory, TOPReward provides scalable, annotation-free reward estimation for robot learning and data curation.

The codebase supports two prediction methods:
- **TOPReward**: Computes log-likelihood rewards for instruction matching on video trajectories (the TOPReward method)
- **GVL**: Generative Value Learning - predicts task completion percentages (0-100%) from shuffled video frames

---

## News

- **02-22-2026**: TOPReward is available on [arXiv](https://arxiv.org/pdf/2602.19313).

---

## Table of Contents

- [TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics](#topreward-token-probabilities-as-hidden-zero-shot-rewards-for-robotics)
  - [News](#news)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
    - [GVL Prediction](#gvl-prediction)
    - [TOPReward Prediction](#topreward-prediction)
    - [Single Runner Script](#single-runner-script)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Environment Variables](#environment-variables)
  - [Configuration (Hydra)](#configuration-hydra)
  - [Extending TOPReward](#extending-topreward)
    - [Adding a New Model](#adding-a-new-model)
    - [Adding a New Dataset](#adding-a-new-dataset)
  - [Troubleshooting](#troubleshooting)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)
  - [License](#license)

---

## Quick Start

After setup (see Getting Started), run either prediction mode:

### GVL Prediction

```bash
HYDRA_FULL_ERROR=1 PYTHONPATH=. uv run python3 -m topreward.scripts.predict \
  --config-dir configs/experiments \
  --config-name predict_gvl
```

### TOPReward Prediction

```bash
HYDRA_FULL_ERROR=1 PYTHONPATH=. uv run python3 -m topreward.scripts.predict \
  --config-dir configs/experiments \
  --config-name predict_topreward \
  model=qwen
```

### Single Runner Script

If you prefer one shell entrypoint, use:

```bash
topreward/scripts/run_predict.sh --config-name predict_gvl dataset=nyudoor model=gemini
topreward/scripts/run_predict.sh --config-name predict_topreward dataset=austin_sirius_dataset model=qwen prediction.add_chat_template=true
```

The script selects an experiment config by name and forwards all remaining arguments as Hydra overrides.

Common override types:
- Group selection: `dataset=... model=... data_loader=... mapper=... prompts=...`
- Scalar values: `prediction.num_examples=20 prediction.output_dir=./results model.model_name=...`
- Booleans: `prediction.add_chat_template=true prediction.use_video_description=false`

Results are saved under `outputs/DATE_TIME/` with predictions, raw outputs, and metrics.

Tip: you can override any config at the CLI, e.g. `model.temperature=0.5`.

---

## Getting Started

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for environment and dependency management
- `ffmpeg` available on your system PATH

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/TOPReward/TOPReward.git
   cd TOPReward
   ```

2. Install `ffmpeg` (if not already installed):
   ```bash
   # macOS (Homebrew)
   brew install ffmpeg

   # Ubuntu / Debian
   sudo apt-get update && sudo apt-get install -y ffmpeg
   ```

3. Set up a `uv` virtual environment and install dependencies:
    ```bash
    uv venv
    source .venv/bin/activate
    uv sync
    ```

### Environment Variables
Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Then edit `.env` with your credentials:
```
OPENAI_API_KEY="your-openai-api-key"
GOOGLE_API_KEY="your-google-api-key"
HUGGING_FACE_HUB_TOKEN="your-hugging-face-token"
```

---

## Configuration (Hydra)

Configuration lives in `configs/`:
- `configs/model/`: model configs (e.g., `gemini.yaml`, `gemma.yaml`, `openai.yaml`)
- `configs/dataset/`: dataset configs
- `configs/data_loader/`: data loader configs (e.g., `huggingface.yaml`, `local.yaml`)
- `configs/prompts/`: prompt styles
- `configs/experiments/`: complete experiment presets (e.g., `predict_gvl.yaml`)

Override parameters from the command line. Examples:
```bash
# Run with explicit experiment config
PYTHONPATH=. uv run python3 -m topreward.scripts.predict --config-dir configs/experiments --config-name predict_gvl

# Override individual fields
PYTHONPATH=. uv run python3 -m topreward.scripts.predict --config-dir configs/experiments --config-name predict_gvl \
  model=gemini dataset=berkeleymvp data_loader=huggingface model.temperature=0.5
```

Run TOPReward on a single local video directly (no frame extraction step):
```bash
topreward/scripts/run_predict.sh --config-name predict_topreward \
  data_loader=local \
  dataset=local_video \
  dataset.video_path=/absolute/path/to/video.mp4 \
  dataset.instruction="open the drawer" \
  prediction.num_examples=1 \
  prediction.output_dir=./results/local_video_topreward
```


---

## Extending TOPReward

### Adding a New Model

TOPReward clients inherit from `topreward.clients.base.BaseModelClient`. You only need to implement `_generate_from_events(self, events: list[Event]) -> str`, which receives a provider-agnostic sequence of text/image events already assembled by the framework. See `topreward/clients/gemini.py` for a complete reference implementation.

1) Implement a client in `topreward/clients/my_model.py`:

```python
# topreward/clients/my_model.py (concise example)
import os
from typing import cast, List

from loguru import logger

from topreward.clients.base import BaseModelClient
from topreward.utils.aliases import Event, ImageEvent, ImageT, TextEvent
from topreward.utils.images import encode_image


class MyModelClient(BaseModelClient):
  def __init__(self, *, rpm: float = 0.0, model_name: str):
    super().__init__(rpm=rpm)
    if not os.getenv("MY_MODEL_API_KEY"):
      raise OSError("Missing MY_MODEL_API_KEY")
    self.model_name = model_name
    logger.info(f"Using MyModel '{self.model_name}'")

  def _generate_from_events(self, events: List[Event]) -> str:
    parts: List[bytes | str] = []
    for ev in events:
      if isinstance(ev, TextEvent):
        parts.append(ev.text)
      elif isinstance(ev, ImageEvent):
        parts.append(encode_image(cast(ImageT, ev.image)))

    # Call your provider with `parts` and return the provider's text response.
    # Placeholder response for docs/tests:
    return "Frame 1: Task Completion: 50%\nFrame 2: Task Completion: 100%"
```

2) Add a Hydra config at `configs/model/my_model.yaml`:

```yaml
_target_: topreward.clients.my_model.MyModelClient
model_name: my-model-name
rpm: 15  # requests per minute (rate limiter)
```

3) Use your model via CLI or experiment config:

```bash
PYTHONPATH=. uv run python3 -m topreward.scripts.predict \
  --config-dir configs/experiments \
  --config-name predict_gvl \
  model=my_model
```
---

### Adding a New Dataset

Create a dataset config that matches the keys used by our HuggingFace loader (`configs/data_loader/huggingface.yaml`). Example:

```yaml
# configs/dataset/my_dataset.yaml
name: my_dataset
dataset_name: "org-or-user/my_dataset_on_hub"
camera_index: 0
max_episodes: 100
num_frames: 15
num_context_episodes: 2
```

Then choose a loader (e.g., Hugging Face) in your experiment or via CLI:

```bash
PYTHONPATH=. uv run python3 -m topreward.scripts.predict \
  --config-dir configs/experiments \
  --config-name predict_gvl \
  dataset=my_dataset data_loader=huggingface
```

---

## Troubleshooting

- macOS library path:
  ```bash
  export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
  ```
- GPU OOM (CUDA): reduce `batch_size` or image resolution in the model config (e.g., `configs/model/gemini.yaml`).
- Hugging Face authentication: ensure `HUGGING_FACE_HUB_TOKEN` is set in `.env` for gated models/private datasets.
- API rate limits: consider lowering concurrency or increasing `TQDM_MININTERVAL` when applicable.

---

## Citation

If you use TOPReward in your research, please cite:

```bibtex
@article{chen2026topreward,
  title={TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics},
  author={Chen, Shirui and Harrison, Cole and Lee, Ying-Chun and Yang, Angela Jin and Ren, Zhongzheng and Ratliff, Lillian J and Duan, Jiafei and Fox, Dieter and Krishna, Ranjay},
  journal={arXiv preprint arXiv:2602.19313},
  year={2026}
}
```

---

## Acknowledgements

TOPReward builds on [OpenGVL / GVL (Generative Value Learning)](https://github.com/budzianowski/opengvl).
We reuse and adapt substantial portions of that implementation throughout this repository, and thank the OpenGVL authors:
Paweł Budzianowski, Emilia Wiśnios, Gracjan Góral, Michał Tyrolski, Igor Kulakov, Viktor Petrenko, and Krzysztof Walas.

Video processing utilities in `topreward/utils/video_utils.py` are adapted from
[LeRobot](https://github.com/huggingface/lerobot) (HuggingFace), licensed under Apache 2.0.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
