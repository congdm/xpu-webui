# xpu-webui

A simple web UI for running [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) on Intel XPU (e.g. Intel Arc B580 12 GB) using native PyTorch XPU support.

## Requirements

- Intel Arc GPU (e.g. B580) with up-to-date drivers
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

## Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| Prompt | — | Text description of the image to generate |
| Negative prompt | — | Features to avoid in the output |
| Width / Height | 1024 | Output resolution (multiples of 64, max 1536) |
| Inference steps | 4 | Z-Image-Turbo is a turbo model — 4 steps is recommended |
| Guidance scale | 0.0 | Classifier-free guidance weight (0 = disabled, as intended for turbo models) |
| Seed | -1 | Fixed seed for reproducibility; -1 uses a random seed |

## Model

**Z-Image-Turbo** (`Tongyi-MAI/Z-Image-Turbo`) is a turbo-distilled SDXL text-to-image model that produces high-quality 1024×1024 images in as few as 4 inference steps.
