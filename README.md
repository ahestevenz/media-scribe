# media-scribe

A Python framework for generating **text**, **images**, and **short videos** by combining large language models with diffusion models. Built on Meta's Llama 3 and the Stable Diffusion family via HuggingFace Transformers and Diffusers.

---

## Features

- **Interactive text generation** via Llama 3 (8B Instruct) with configurable chat history, temperature, and sampling
- **LLaMA-assisted prompt refinement** — chat with Llama 3 to iteratively improve your image prompt before generation
- **Image generation** with Stable Diffusion 3, SDXL, CivitAI models, and pix2pix
- **Image-to-image** transformation using InstructPix2Pix and SD 1.5 img2img pipelines
- **Short video generation** — produce an animated WebP from a sequence of prompts (each prompt = one frame)
- **Unified YAML config** for all model settings
- **MPS (Apple Silicon), CUDA, and CPU** device support

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python >= 3.10 | Tested on 3.11 |
| PyTorch >= 2.4 | MPS (Apple Silicon), CUDA, or CPU |
| HuggingFace account | Required for gated models (Llama 3) |

> **Llama 3 access**: Request access at [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), accept the licence, then authenticate locally:
> ```bash
> huggingface-cli login
> ```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/ahestevenz/media-scribe.git
cd media-scribe

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install the package
pip install -e .

# 4. (Optional) Install dev tools including pre-commit hooks
pip install -e ".[dev]"
pre-commit install
```

---

## Configuration

Edit `config.yml` before first use. The key fields:

```yaml
verbose: true
device: "mps"           # "mps" | "cuda" | "cpu"

llama_config:
  model_name: "meta-llama/Meta-Llama-3-8B-Instruct"
  system_prompt: "You are a helpful assistant"
  max_num_historical_messages: 6
  max_tokens: 256
  temperature: 0.7
  top_p: 0.9

sd_config:
  root_models_path: ~/.cache/huggingface/hub   # absolute path to your HF cache
  root_output_dir: ~/Desktop/media-scribe-outputs
  model_type: "sd_xl"   # active model — see Supported Models below
  load_refiner: true
  num_inference_steps: 50
  guidance_scale: 7.5
  model_paths:
    sd_3:
      - models--stabilityai--stable-diffusion-3-medium-diffusers/sd3_medium_incl_clips_t5xxlfp16.safetensors
      - null
    sd_xl:
      - models--Stable-Diffusion-XL/sd_xl_base_1.0.safetensors
      - models--Stable-Diffusion-XL/sd_xl_refiner_1.0.safetensors
    civitai:
      - models--civitai/juggernautXL_juggXIByRundiffusion.safetensors
      - models--Stable-Diffusion-XL/sd_xl_refiner_1.0.safetensors
    pix_2_pix:
      - models--timbrooks--instruct-pix2pix/snapshots/<commit-hash>
      - null
    sd_1_5_img_2_img:
      - models--radames--stable-diffusion-v1-5-img2img/snapshots/<commit-hash>
      - null
```

Each entry in `model_paths` is a two-element list `[base_model, refiner]`. Use `null` for the refiner when the model does not have one. All paths are relative to `root_models_path`.

---

## Usage

### CLI

#### Text generation

Opens an interactive Llama 3 session:

```bash
bn-run-scribe-text -c config.yml
```

```
You: Write me a haiku about diffusion models
LLaMA:
 Noise fades to signal,
 Guided by a gentle hand—
 Art blooms from chaos.
You: exit
```

#### Image generation

By default, a LLaMA chat session opens first so you can refine your prompt interactively. Type `exit` when satisfied and image generation starts:

```bash
bn-run-scribe-image -c config.yml
```

Skip the LLaMA step and type a prompt directly:

```bash
bn-run-scribe-image -c config.yml --no-llama
```

Image-to-image (requires `pix_2_pix` or `sd_1_5_img_2_img` as `model_type` in config):

```bash
bn-run-scribe-image -c config.yml --no-llama \
  --path-image path/to/source.png \
  --strength 0.75
```

Full flag reference:

| Flag | Short | Default | Description |
|---|---|---|---|
| `--conf` | `-c` | required | Path to YAML config file |
| `--no-llama` | `-nl` | false | Skip LLaMA prompt refinement |
| `--path-image` | `-i` | none | Source image for img2img |
| `--strength` | `-s` | 0.75 | img2img denoising strength (0–1) |
| `--verbose` | `-v` | INFO | Repeat for DEBUG (`-vv`) |
| `--profile` | `-p` | none | Write cProfile output to file |

### Python API

```python
from media_scribe.media_scribe_config import MediaScribeConfig
from media_scribe.llama_text_scribe import LlamaTextScribe
from media_scribe.image_video_scribe import ImageVideoScribe

config = MediaScribeConfig.from_yaml("config.yml")

# Text generation
llama = LlamaTextScribe(config)
response = llama.generate_text("Explain diffusion models in simple terms")
print(response)

# Image generation — returns Path to the saved PNG
scribe = ImageVideoScribe(config)
image_path = scribe.generate_image(
    prompt="A futuristic city at sunset, cinematic lighting",
    negative_prompt="blurry, low quality",
)

# Image-to-image — requires pix_2_pix or sd_1_5_img_2_img model type
from pathlib import Path
image_path = scribe.generate_image_from_image(
    prompt="Turn this into a watercolour painting",
    img_path=Path("source.png"),
    strength=0.75,
)

# Video generation — returns Path to an animated WebP.
# Individual frames are also saved as frame_000.png, frame_001.png, ...
# Only works with text-to-image model types (sd_3, sd_xl, civitai).
video_path = scribe.generate_video(
    prompts=[
        "a lone wolf standing on a snowy hill at dusk",
        "the wolf lifts its head and howls",
        "the wolf turns and disappears into the forest",
    ],
    fps=8,
    num_inference_steps=20,  # fewer steps per frame is fine for video
)
```

---

## Supported Models

| Type | Model | `model_type` key |
|---|---|---|
| LLM | Meta Llama 3 8B Instruct | — |
| Text-to-image | Stable Diffusion 3 Medium | `sd_3` |
| Text-to-image | Stable Diffusion XL Base + Refiner | `sd_xl` |
| Text-to-image | JuggernautXL (CivitAI) | `civitai` |
| Image-to-image | InstructPix2Pix | `pix_2_pix` |
| Image-to-image | SD 1.5 img2img | `sd_1_5_img_2_img` |

Text-to-image models support `generate_image()` and `generate_video()`.
Image-to-image models support `generate_image_from_image()` only.

---

## Project Structure

```
media-scribe/
├── src/
│   └── media_scribe/
│       ├── __init__.py
│       ├── image_video_scribe.py   # ImageVideoScribe
│       ├── llama_text_scribe.py    # LlamaTextScribe
│       ├── media_scribe_config.py  # Pydantic config models
│       ├── utils.py                # Interactive CLI loop
│       └── scripts/
│           ├── run_scribe_image.py
│           └── run_scribe_text.py
├── config.yml
├── pyproject.toml
├── .pre-commit-config.yaml
└── README.md
```

---

## Development

```bash
# Run all linters and type checks
pre-commit run --all-files
```

The pre-commit pipeline runs: `autopep8`, `flake8` (max line length 120), `mypy`, and `isort`.

---

## License

MIT — see [LICENSE](LICENSE) for details.
