# media-scribe

A Python framework for generating **text**, **images**, and **short videos** by combining large language models with diffusion models. Built on Meta's Llama 3 and the Stable Diffusion family via HuggingFace Transformers and Diffusers.

---

## Features

- **Interactive text generation** via Llama 3 (8B Instruct) with configurable chat history, temperature, and sampling
- **LLaMA-assisted prompt refinement** — chat with Llama 3 to iteratively improve your image prompt before generation
- **Image generation** with Stable Diffusion 3, SDXL, CivitAI models, and pix2pix
- **Image-to-image** transformation using InstructPix2Pix and SD 1.5 img2img pipelines
- **Frame-by-frame video** — produce an animated WebP from a sequence of prompts (each prompt = one frame)
- **SVD video generation** — coherent video from a single text prompt via Stable Video Diffusion (text → anchor image → 14/25-frame clip)
- **Unified YAML config** for all model settings
- **MPS (Apple Silicon), CUDA, and CPU** device support

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python >= 3.10 | Tested on 3.11 |
| PyTorch >= 2.4 | MPS (Apple Silicon), CUDA, or CPU |
| HuggingFace account | Required for gated models (Llama 3, SD3, SVD) |

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

## Downloading Models

All models are stored under `root_models_path` (default `~/.cache/huggingface/hub`). Run `huggingface-cli login` once before downloading any gated model.

```bash
pip install huggingface_hub
huggingface-cli login
```

> **Getting a token:** go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), click **New token**, choose **Read** access, and paste the token when prompted.
>
> **Gated models (Llama 3, SD3, SVD):** a token alone is not enough. You must also visit each model page on HuggingFace and click **Agree and access repository** to accept the licence before the download will succeed.

### 1. Llama 3 8B Instruct *(gated)*

Request access at [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), accept the licence, then:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct
```

The model is loaded by name at runtime via `transformers`; no path entry in `config.yml` is needed.

### 2. Stable Diffusion 3 Medium *(gated)*

Request access at [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers), accept the licence, then:

```bash
huggingface-cli download stabilityai/stable-diffusion-3-medium-diffusers \
  sd3_medium_incl_clips_t5xxlfp16.safetensors \
  --local-dir ~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium-diffusers
```

Config path: `models--stabilityai--stable-diffusion-3-medium-diffusers/sd3_medium_incl_clips_t5xxlfp16.safetensors`

### 3. Stable Diffusion XL Base 1.0

```bash
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 \
  sd_xl_base_1.0.safetensors \
  --local-dir ~/.cache/huggingface/hub/models--Stable-Diffusion-XL
```

Config path: `models--Stable-Diffusion-XL/sd_xl_base_1.0.safetensors`

### 4. Stable Diffusion XL Refiner 1.0

Used as the refiner for `sd_3`, `sd_xl`, and `civitai` model types.

```bash
huggingface-cli download stabilityai/stable-diffusion-xl-refiner-1.0 \
  sd_xl_refiner_1.0.safetensors \
  --local-dir ~/.cache/huggingface/hub/models--Stable-Diffusion-XL
```

Config path: `models--Stable-Diffusion-XL/sd_xl_refiner_1.0.safetensors`

### 5. JuggernautXL by RunDiffusion *(CivitAI)*

This model is distributed via [CivitAI](https://civitai.com/models/133005/juggernaut-xl). Download `juggernautXL_juggXIByRundiffusion.safetensors` from the site, then move it:

```bash
mkdir -p ~/.cache/huggingface/hub/models--civitai
mv ~/Downloads/juggernautXL_juggXIByRundiffusion.safetensors \
   ~/.cache/huggingface/hub/models--civitai/
```

Config path: `models--civitai/juggernautXL_juggXIByRundiffusion.safetensors`

### 6. InstructPix2Pix

```bash
huggingface-cli download timbrooks/instruct-pix2pix
```

The HuggingFace cache manager places this at the exact snapshot path expected by the config automatically.

Config path: `models--timbrooks--instruct-pix2pix/snapshots/31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`

### 7. SD 1.5 img2img

```bash
huggingface-cli download radames/stable-diffusion-v1-5-img2img
```

Config path: `models--radames--stable-diffusion-v1-5-img2img/snapshots/cac24f945bd8f5604e243550bb3092be11964805`

### 8. Stable Video Diffusion img2vid-xt *(gated)*

Used by `bn-run-scribe-video`. Request access at [stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt), accept the licence, then:

```bash
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt
```

The pipeline is downloaded automatically on first run if `svd_model_path` is `null` in `config.yml`. To pin a local copy, set:

```yaml
sd_config:
  svd_model_path: null   # or an absolute/relative local path
```

---

## Configuration

Copy `config.example.yml` to `config.yml` and edit the key fields:

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
  root_models_path: ~/.cache/huggingface/hub
  root_output_dir: ~/Desktop/media-scribe-outputs
  model_type: "sd_xl"   # active model — see Supported Models below
  load_refiner: true
  num_inference_steps: 50
  guidance_scale: 7.5
  svd_model_path: null  # null = auto-download SVD from HF on first use
  model_paths:
    sd_3:
      - models--stabilityai--stable-diffusion-3-medium-diffusers/sd3_medium_incl_clips_t5xxlfp16.safetensors
      - models--Stable-Diffusion-XL/sd_xl_refiner_1.0.safetensors
    sd_xl:
      - models--Stable-Diffusion-XL/sd_xl_base_1.0.safetensors
      - models--Stable-Diffusion-XL/sd_xl_refiner_1.0.safetensors
    civitai:
      - models--civitai/juggernautXL_juggXIByRundiffusion.safetensors
      - models--Stable-Diffusion-XL/sd_xl_refiner_1.0.safetensors
    pix_2_pix:
      - models--timbrooks--instruct-pix2pix/snapshots/31519b5cb02a7fd89b906d88731cd4d6a7bbf88d
      - null
    sd_1_5_img_2_img:
      - models--radames--stable-diffusion-v1-5-img2img/snapshots/cac24f945bd8f5604e243550bb3092be11964805
      - null
```

Each entry in `model_paths` is `[base_model, refiner]`. Use `null` for the refiner when the model does not have one. All paths are relative to `root_models_path`. Tilde (`~`) is expanded automatically.

---

## Usage

### CLI

#### Text generation

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

By default, a LLaMA chat session opens first so you can refine your prompt. Type `exit` when satisfied and generation starts:

```bash
bn-run-scribe-image -c config.yml
```

Skip LLaMA and type a prompt directly:

```bash
bn-run-scribe-image -c config.yml --no-llama
```

Image-to-image (requires `pix_2_pix` or `sd_1_5_img_2_img` as `model_type` in config):

```bash
bn-run-scribe-image -c config.yml --no-llama \
  --path-image path/to/source.png \
  --strength 0.75
```

| Flag | Short | Default | Description |
|---|---|---|---|
| `--conf` | `-c` | required | Path to YAML config file |
| `--no-llama` | `-nl` | false | Skip LLaMA prompt refinement |
| `--path-image` | `-i` | none | Source image for img2img |
| `--strength` | `-s` | 0.75 | img2img denoising strength (0–1) |
| `--verbose` | `-v` | INFO | Repeat for DEBUG (`-vv`) |
| `--profile` | `-p` | none | Write cProfile output to file |

#### SVD video generation

Generates a temporally coherent video from a single text prompt. Internally: prompt → anchor image (base model) → 14/25-frame video (Stable Video Diffusion).

```bash
bn-run-scribe-video -c config.yml --no-llama
```

With LLaMA-assisted prompt refinement:

```bash
bn-run-scribe-video -c config.yml
```

| Flag | Short | Default | Description |
|---|---|---|---|
| `--conf` | `-c` | required | Path to YAML config file |
| `--no-llama` | `-nl` | false | Skip LLaMA prompt refinement |
| `--num-frames` | `-n` | 25 | Frames to generate (14 or 25 for SVD) |
| `--fps` | | 7 | Output frames per second |
| `--motion-bucket-id` | | 127 | Motion amount (0–255); higher = more motion |
| `--noise-aug-strength` | | 0.02 | Anchor frame noise (0.0–1.0) |
| `--num-inference-steps` | | 25 | SVD denoising steps |
| `--decode-chunk-size` | | 4 | VAE frames decoded at once; lower = less peak memory |
| `--verbose` | `-v` | INFO | Repeat for DEBUG (`-vv`) |
| `--profile` | `-p` | none | Write cProfile output to file |

> **Memory note (MPS / Apple Silicon):** SVD is memory-intensive. On MPS the base model is automatically offloaded to CPU before SVD runs, and the anchor image is resized to 512×320 (instead of 1024×576) to fit within device limits. If you still run out of memory, lower `--num-frames` to 14 and `--decode-chunk-size` to 2.

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

# Frame-by-frame video — each prompt becomes one frame, returns animated WebP
video_path = scribe.generate_video(
    prompts=[
        "a lone wolf standing on a snowy hill at dusk",
        "the wolf lifts its head and howls",
        "the wolf turns and disappears into the forest",
    ],
    fps=8,
    num_inference_steps=20,
)

# SVD video — single prompt → anchor image → coherent video clip
video_path = scribe.generate_video_svd(
    prompt="A majestic eagle soaring over mountain peaks at golden hour",
    negative_prompt="blurry, low quality",
    num_frames=14,       # 14 or 25
    fps=7,
    motion_bucket_id=127,
    decode_chunk_size=4, # lower to reduce peak memory on MPS
)
```

---

## Supported Models

| Type | Model | `model_type` key | Gated |
|---|---|---|---|
| LLM | Meta Llama 3 8B Instruct | — | yes |
| Text-to-image | Stable Diffusion 3 Medium | `sd_3` | yes |
| Text-to-image | Stable Diffusion XL Base + Refiner | `sd_xl` | no |
| Text-to-image | JuggernautXL (CivitAI) | `civitai` | no |
| Image-to-image | InstructPix2Pix | `pix_2_pix` | no |
| Image-to-image | SD 1.5 img2img | `sd_1_5_img_2_img` | no |
| Text-to-video | Stable Video Diffusion img2vid-xt | SVD (via `generate_video_svd`) | yes |

Text-to-image models (`sd_3`, `sd_xl`, `civitai`) support `generate_image()`, `generate_video()`, and `generate_video_svd()`.
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
│           ├── run_scribe_text.py
│           └── run_scribe_video.py
├── config.example.yml
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
