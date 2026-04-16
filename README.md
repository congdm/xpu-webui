# xpu-webui

A Gradio web UI for [Z-Image-Turbo](https://huggingface.co/Comfy-Org/z_image_turbo) on Intel XPU (e.g. Intel Arc B580) using **native PyTorch XPU support** (no IPEX required).

## Architecture

| Component | Source | Implementation |
|-----------|--------|----------------|
| Transformer | `Comfy-Org/z_image_turbo` (single `.safetensors`) | `modules/transformer.py` — self-contained nn modules ported from [diffusers ZImagePipeline](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/z_image) |
| Scheduler | — | `modules/scheduler.py` — `FlowMatchEulerDiscreteScheduler` with exponential shift |
| Text encoder | `Tongyi-MAI/Z-Image-Turbo` | Qwen3 loaded from local `.safetensors` with `transformers.AutoModel.from_config(...)` + `load_state_dict(...)` |
| VAE | `Tongyi-MAI/Z-Image-Turbo` | `AutoencoderKL` loaded from local `.safetensors` + local config |

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
All model files must exist locally before launch.

For low-VRAM GPUs, runtime stages are loaded on-demand during generation:
1. Load tokenizer + text encoder, encode prompt, unload both.
2. Transformer stage (preloaded to CPU at startup):
   - Block offload ON: stream blocks CPU→GPU during denoising, no GPU memory for full model.
   - Block offload OFF: copy full transformer to GPU, denoise, move back to CPU after.
This reduces peak VRAM usage at the cost of longer per-image latency.

Expected local layout:

```text
models/
	diffusion_models/
		z_image_turbo_bf16.safetensors
	text_encoders/
		qwen_3_4b.safetensors
		config.json                # optional; built-in fallback exists for Z-Image-Turbo
		tokenizer.json
		tokenizer_config.json
		merges.txt
		vocab.json
	vae/
		ae.safetensors
		config.json                # optional; built-in fallback exists for Z-Image-Turbo
```

The tokenizer files (tokenizer.json, tokenizer_config.json, merges.txt, vocab.json) come from the upstream `tokenizer/` folder (https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/main/tokenizer). The text encoder and VAE weights are loaded directly from local `.safetensors` files rather than via `from_pretrained(...)` weight loading.

## Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| Prompt | — | Text description of the image to generate |
| Negative prompt | — | Features to suppress (only used when Guidance scale > 0) |
| Width / Height | 1024 | Output resolution; must be multiples of 16, max 1536 |
| Inference steps | 9 | Recommended by upstream; more steps = higher quality |
| Guidance scale | 0.0 | CFG weight — 0 = turbo mode (no classifier-free guidance) |
| Transformer block offload | On | Runs transformer layers one-by-one from CPU to GPU for lower VRAM; slower |
| Seed | -1 | Fixed seed for reproducibility; -1 uses a random seed |

## Model

**Z-Image-Turbo** is a flow-matching text-to-image model by Alibaba Tongyi.  
The transformer uses a single-stream architecture with adaLN modulation, RoPE position embeddings, and a Qwen3 text encoder.

