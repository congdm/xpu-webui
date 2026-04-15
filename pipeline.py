"""
pipeline.py – Z-Image-Turbo inference on Intel XPU using native PyTorch XPU support.

Model: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
"""

import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"

# Resolve the best available device: XPU → CUDA → CPU
def _get_device() -> str:
    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pipeline() -> tuple[StableDiffusionXLPipeline, str]:
    """Load the Z-Image-Turbo pipeline onto the best available device."""
    device = _get_device()

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    # Euler Ancestral scheduler works well for turbo-distilled models
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config
    )

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    return pipe, device


def generate(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
    guidance_scale: float = 0.0,
    seed: int = -1,
):
    """
    Run a single inference pass and return a PIL image.

    Z-Image-Turbo is a turbo-distilled model:
    - Very few steps (4 is the recommended default).
    - Classifier-free guidance is collapsed (guidance_scale=0.0).
    """
    generator = None
    if seed >= 0:
        generator = torch.Generator().manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return result.images[0]
