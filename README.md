# xpu-webui

A Gradio web UI for [Z-Image-Turbo](https://huggingface.co/Comfy-Org/z_image_turbo) on Intel XPU (e.g. Intel Arc B580) using **native PyTorch XPU support** (no IPEX required).

## Architecture

| Component | Source | Implementation |
|-----------|--------|----------------|
| Transformer | `Comfy-Org/z_image_turbo` (single `.safetensors`) | `modules/transformer.py` — self-contained nn modules ported from [diffusers ZImagePipeline](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/z_image) |
| Scheduler | — | `modules/scheduler.py` — `FlowMatchEulerDiscreteScheduler` with exponential shift |
| Text encoder | `Tongyi-MAI/Z-Image-Turbo` | Qwen3 via `transformers.AutoModel` |
| VAE | `Tongyi-MAI/Z-Image-Turbo` | `diffusers.AutoencoderKL` |

## Requirements

- Intel Arc GPU with up-to-date drivers
- Python 3.10+

## Installation

### 1. Install PyTorch with native XPU support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

### 2. Install remaining dependencies

```bash
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

Open your browser at <http://localhost:7860>.  
Model weights are downloaded automatically on first launch (~15 GB total).

## Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| Prompt | — | Text description of the image to generate |
| Negative prompt | — | Features to suppress (only used when Guidance scale > 0) |
| Width / Height | 1024 | Output resolution; must be multiples of 16, max 1536 |
| Inference steps | 9 | Recommended by upstream; more steps = higher quality |
| Guidance scale | 0.0 | CFG weight — 0 = turbo mode (no classifier-free guidance) |
| Seed | -1 | Fixed seed for reproducibility; -1 uses a random seed |

## Model

**Z-Image-Turbo** is a flow-matching text-to-image model by Alibaba Tongyi.  
The transformer uses a single-stream architecture with adaLN modulation, RoPE position embeddings, and a Qwen3 text encoder.

