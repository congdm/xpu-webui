"""
pipeline.py – Z-Image-Turbo inference on Intel XPU.

Model weights  : models/diffusion_models/  (ComfyUI single-file .safetensors)
Text encoder   : models/text_encoders/     (Qwen3 tokenizer/encoder)
VAE            : models/vae/               (AutoencoderKL)

Core transformer and scheduler are implemented in modules/ (no diffusers pipeline class).
"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from modules import FlowMatchEulerDiscreteScheduler, ZImageTransformer2DModel

# ── Local model directories ────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).parent
DIFFUSION_MODELS_DIR = _BASE_DIR / "models" / "diffusion_models"
TEXT_ENCODERS_DIR    = _BASE_DIR / "models" / "text_encoders"
VAE_DIR              = _BASE_DIR / "models" / "vae"

# Transformer default config (matches Tongyi-MAI/Z-Image-Turbo model_index.json)
TRANSFORMER_CONFIG = dict(
    all_patch_size=(2,),
    all_f_patch_size=(1,),
    in_channels=16,
    dim=3840,
    n_layers=30,
    n_refiner_layers=2,
    n_heads=30,
    n_kv_heads=30,
    norm_eps=1e-5,
    qk_norm=True,
    cap_feat_dim=2560,
    rope_theta=256.0,
    t_scale=1000.0,
    axes_dims=[32, 48, 48],
    axes_lens=[1024, 512, 512],
)

MAX_SEQ_LEN = 512
VAE_SCALE_FACTOR = 8   # 2^(num_vae_blocks-1) where num_vae_blocks=4 → 2^3 = 8


# ── Device selection ───────────────────────────────────────────────────────────

def _get_device() -> str:
    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Checkpoint loading ─────────────────────────────────────────────────────────

def _find_local_safetensors(directory: Path) -> Path:
    """Return the first .safetensors file found in the given local directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Model directory not found: {directory}")
    for path in sorted(directory.iterdir()):
        if path.suffix == ".safetensors":
            return path
    raise FileNotFoundError(f"No .safetensors file found in {directory}")


def _load_transformer(device: str) -> ZImageTransformer2DModel:
    """Load the ComfyUI single-file checkpoint from local disk into our nn module."""
    ckpt_path = _find_local_safetensors(DIFFUSION_MODELS_DIR)
    print(f"  Transformer checkpoint: {ckpt_path.name}")

    state_dict = load_file(ckpt_path, device="cpu")

    # Strip common ComfyUI wrapper prefixes if present
    for prefix in ("model.", "transformer."):
        if all(k.startswith(prefix) for k in state_dict):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            break

    model = ZImageTransformer2DModel(**TRANSFORMER_CONFIG)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys in transformer checkpoint")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys in transformer checkpoint")

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


# ── Timestep / shift helpers (ported from diffusers pipeline_z_image.py) ──────

def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


# ── Text encoding ──────────────────────────────────────────────────────────────

def _encode_prompt(
    tokenizer: AutoTokenizer,
    text_encoder: AutoModel,
    prompt: str,
    device: str,
    max_seq_len: int = MAX_SEQ_LEN,
) -> torch.Tensor:
    """
    Encode a single prompt string → variable-length embedding tensor [S, D].

    The Qwen3 encoder's second-to-last hidden state is used, masked to the
    actual (non-padding) token positions.
    """
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    inputs = tokenizer(
        [prompt_text],
        padding="max_length",
        max_length=max_seq_len,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        hidden = text_encoder(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
        ).hidden_states[-2]   # second-to-last layer

    mask = inputs.attention_mask[0].bool()
    return hidden[0][mask]   # [num_real_tokens, D]


# ── VAE scale factors (read from VAE config) ───────────────────────────────────

def _latent_size(height: int, width: int, vae_scale: int = VAE_SCALE_FACTOR) -> tuple[int, int]:
    """Image → latent spatial dimensions (factor of 2 rounding for ZImage)."""
    lat_h = 2 * (height // (vae_scale * 2))
    lat_w = 2 * (width // (vae_scale * 2))
    return lat_h, lat_w


# ── Public API ─────────────────────────────────────────────────────────────────

@dataclass
class PipelineComponents:
    transformer: ZImageTransformer2DModel
    tokenizer: AutoTokenizer
    text_encoder: AutoModel
    vae: AutoencoderKL
    scheduler: FlowMatchEulerDiscreteScheduler
    device: str


def load_pipeline() -> tuple[PipelineComponents, str]:
    """Load all pipeline components from local model directories."""
    device = _get_device()
    print(f"Device: {device}")

    print("Loading transformer…")
    transformer = _load_transformer(device)

    print("Loading tokenizer + text encoder…")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODERS_DIR)
    text_encoder = AutoModel.from_pretrained(
        TEXT_ENCODERS_DIR,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    print("Loading VAE…")
    vae = AutoencoderKL.from_pretrained(
        VAE_DIR,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

    components = PipelineComponents(
        transformer=transformer,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        device=device,
    )
    return components, device


@torch.no_grad()
def generate(
    components: PipelineComponents,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 9,
    guidance_scale: float = 0.0,
    seed: int = -1,
) -> Image.Image:
    """
    Run Z-Image-Turbo text-to-image inference and return a PIL image.

    Args:
        components:      Output of load_pipeline().
        prompt:          Text description of the desired image.
        negative_prompt: Features to suppress (only active when guidance_scale > 0).
        width/height:    Output resolution; must be multiples of 16.
        steps:           Denoising steps (9 is the upstream recommendation).
        guidance_scale:  CFG weight; 0.0 disables classifier-free guidance.
        seed:            RNG seed (-1 = random).
    """
    device = components.device

    # ── Validate resolution
    vae_scale = VAE_SCALE_FACTOR * 2  # = 16
    if height % vae_scale != 0 or width % vae_scale != 0:
        raise ValueError(f"height and width must be multiples of {vae_scale}.")

    # ── Seed
    generator = torch.Generator(device=device)
    if seed >= 0:
        generator.manual_seed(seed)
    else:
        generator.seed()

    # ── Encode text
    pos_embeds = _encode_prompt(components.tokenizer, components.text_encoder, prompt, device)
    do_cfg = guidance_scale > 0.0
    if do_cfg:
        neg_prompt = negative_prompt if negative_prompt else ""
        neg_embeds = _encode_prompt(components.tokenizer, components.text_encoder, neg_prompt, device)

    # ── Prepare latents [1, C, F=1, H', W']
    lat_h, lat_w = _latent_size(height, width)
    # Latents initialised in float32 for numerical stability; cast to transformer
    # dtype (bfloat16) only inside the denoising loop, matching the upstream pipeline.
    latents = torch.randn(1, TRANSFORMER_CONFIG["in_channels"], lat_h, lat_w,
                          device=device, dtype=torch.float32, generator=generator)

    # ── Scheduler
    image_seq_len = (lat_h // 2) * (lat_w // 2)
    mu = _calculate_shift(
        image_seq_len,
        **{k: v for k, v in components.scheduler.config.items()},
    )
    components.scheduler.sigma_min = 0.0
    components.scheduler.set_timesteps(steps, device=device, mu=mu)
    timesteps = components.scheduler.timesteps

    # ── Denoising loop
    for t in timesteps:
        timestep = t.expand(latents.shape[0])
        t_norm = (1000 - timestep) / 1000      # normalise to [0, 1]

        if do_cfg:
            # Duplicate for positive + negative
            lat_in = latents.to(components.transformer.dtype)
            lat_in = lat_in.unsqueeze(2)       # [1, C, 1, H', W']
            lat_list = list(lat_in.repeat(2, 1, 1, 1, 1).unbind(dim=0))
            cap_list = [pos_embeds, neg_embeds]
            t_in = t_norm.repeat(2)
        else:
            lat_in = latents.to(components.transformer.dtype).unsqueeze(2)
            lat_list = list(lat_in.unbind(dim=0))
            cap_list = [pos_embeds]
            t_in = t_norm

        model_out = components.transformer(
            lat_list, t_in, cap_list, return_dict=False
        )[0]  # list of [C, 1, H', W']

        if do_cfg:
            pos_out = model_out[0].float()
            neg_out = model_out[1].float()
            noise_pred = pos_out + guidance_scale * (pos_out - neg_out)
        else:
            noise_pred = model_out[0].float()

        # Squeeze temporal dim and negate (convention in ZImage pipeline)
        noise_pred = noise_pred.squeeze(1).unsqueeze(0)   # [1, C, H', W']
        noise_pred = -noise_pred

        latents = components.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # ── VAE decode
    latents = latents.to(components.vae.dtype)
    scaling = components.vae.config.scaling_factor
    shift   = getattr(components.vae.config, "shift_factor", 0.0)
    latents = (latents / scaling) + shift

    image_tensor = components.vae.decode(latents, return_dict=False)[0]
    # image_tensor: [1, 3, H, W] in [-1, 1] (standard VAE output)
    image_tensor = (image_tensor.float().clamp(-1, 1) + 1) / 2   # → [0, 1]
    image_np = (image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(image_np)
