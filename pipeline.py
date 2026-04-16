"""
pipeline.py – Z-Image-Turbo inference on Intel XPU.

Model weights  : models/diffusion_models/  (ComfyUI single-file .safetensors)
Text encoder   : models/text_encoders/     (Qwen3 tokenizer/encoder)
VAE            : models/vae/               (AutoencoderKL)

Core transformer and scheduler are implemented in modules/ (no diffusers pipeline class).
"""

import math
import json
import gc
import os
import time
import contextlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from modules import (
    FlowMatchEulerDiscreteScheduler,
    ZImageTransformer2DModel,
    attention_backend,
    attention_chunk_size,
)

# ── Local model directories ────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).parent
DIFFUSION_MODELS_DIR = _BASE_DIR / "models" / "diffusion_models"
TEXT_ENCODERS_DIR    = _BASE_DIR / "models" / "text_encoders"
TOKENIZER_DIR        = _BASE_DIR / "models" / "tokenizer"
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
TEXT_ENCODER_CHUNK_SIZE = 128
TEXT_ENCODER_ATTN_IMPL = "qwen3_chunked"

TEXT_ENCODER_CONFIG = {
    "architectures": ["Qwen3Model"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 9728,
    "max_position_embeddings": 40960,
    "max_window_layers": 36,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-6,
    "rope_scaling": None,
    "rope_theta": 1_000_000,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}

VAE_CONFIG = {
    "_class_name": "AutoencoderKL",
    "_diffusers_version": "0.36.0.dev0",
    "_name_or_path": "flux-dev",
    "act_fn": "silu",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ],
    "force_upcast": True,
    "in_channels": 3,
    "latent_channels": 16,
    "latents_mean": None,
    "latents_std": None,
    "layers_per_block": 2,
    "mid_block_add_attention": True,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_size": 1024,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "up_block_types": [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ],
    "use_post_quant_conv": False,
    "use_quant_conv": False,
}

TOKENIZER_REQUIRED_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
)


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


def _resolve_tokenizer_dir() -> Path:
    for directory in (TOKENIZER_DIR, TEXT_ENCODERS_DIR):
        if directory.exists() and all((directory / name).exists() for name in TOKENIZER_REQUIRED_FILES):
            return directory

    required = ", ".join(TOKENIZER_REQUIRED_FILES)
    raise FileNotFoundError(
        "Tokenizer files not found. Expected a local tokenizer directory at "
        f"{TOKENIZER_DIR} or {TEXT_ENCODERS_DIR} containing: {required}."
    )


def _load_tokenizer() -> PreTrainedTokenizerBase:
    tokenizer_dir = _resolve_tokenizer_dir()
    print(f"  Tokenizer directory: {tokenizer_dir.name}")
    return AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)


def _register_qwen_chunked_attention() -> None:
    """Register a chunked Qwen3 attention backend to reduce peak VRAM."""
    from transformers.models.qwen3 import modeling_qwen3

    if TEXT_ENCODER_ATTN_IMPL in modeling_qwen3.ALL_ATTENTION_FUNCTIONS:
        return

    def _chunked_attention_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        # query: [B, H, Sq, Dh], key/value: [B, H_kv, Sk, Dh]
        key_states = key.repeat_interleave(module.num_key_value_groups, dim=1)
        value_states = value.repeat_interleave(module.num_key_value_groups, dim=1)

        bsz, num_heads, seq_len_q, head_dim = query.shape
        out = torch.empty((bsz, num_heads, seq_len_q, head_dim), device=query.device, dtype=query.dtype)

        chunk = TEXT_ENCODER_CHUNK_SIZE
        for start in range(0, seq_len_q, chunk):
            end = min(start + chunk, seq_len_q)
            q = query[:, :, start:end, :]
            attn_scores = torch.matmul(q, key_states.transpose(-2, -1)) * scaling
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask[:, :, start:end, :]

            attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
            if dropout:
                attn_probs = F.dropout(attn_probs, p=dropout, training=module.training)

            out[:, :, start:end, :] = torch.matmul(attn_probs, value_states)

        return out.transpose(1, 2).contiguous(), None

    modeling_qwen3.ALL_ATTENTION_FUNCTIONS.register(TEXT_ENCODER_ATTN_IMPL, _chunked_attention_forward)


def _load_text_encoder(device: str) -> PreTrainedModel:
    _register_qwen_chunked_attention()

    ckpt_path = _find_local_safetensors(TEXT_ENCODERS_DIR)
    print(f"  Text encoder checkpoint: {ckpt_path.name}")

    config_path = TEXT_ENCODERS_DIR / "config.json"
    if config_path.exists():
        config = AutoConfig.from_pretrained(TEXT_ENCODERS_DIR, local_files_only=True)
    else:
        config = AutoConfig.for_model(
            TEXT_ENCODER_CONFIG["model_type"],
            **{k: v for k, v in TEXT_ENCODER_CONFIG.items() if k != "model_type"},
        )

    config._attn_implementation = TEXT_ENCODER_ATTN_IMPL

    # from_pretrained expects canonical names (model.safetensors). Create a hard
    # link when the local checkpoint uses a custom filename.
    hf_ckpt = TEXT_ENCODERS_DIR / "model.safetensors"
    if ckpt_path.name != hf_ckpt.name and not hf_ckpt.exists():
        try:
            os.link(ckpt_path, hf_ckpt)
            print(f"  Linked {hf_ckpt.name} -> {ckpt_path.name}")
        except OSError:
            print("  [warn] Could not create model.safetensors hardlink; using manual loader fallback")

    if hf_ckpt.exists():
        try:
            model = AutoModel.from_pretrained(
                TEXT_ENCODERS_DIR,
                config=config,
                local_files_only=True,
                low_cpu_mem_usage=True,
                attn_implementation=TEXT_ENCODER_ATTN_IMPL,
                torch_dtype=torch.bfloat16,
            )
            model = model.to(device=device, dtype=torch.bfloat16)
            model.eval()
            return model
        except Exception as exc:
            print(f"  [warn] Optimized text encoder loading failed: {exc}")
            print("  [warn] Falling back to manual state_dict loading")

    state_dict = load_file(ckpt_path, device="cpu")
    model = AutoModel.from_config(
        config,
        attn_implementation=TEXT_ENCODER_ATTN_IMPL,
        torch_dtype=torch.bfloat16,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys in text encoder checkpoint")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys in text encoder checkpoint")

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def _load_vae(device: str) -> AutoencoderKL:
    ckpt_path = _find_local_safetensors(VAE_DIR)
    print(f"  VAE checkpoint: {ckpt_path.name}")

    config_dict = deepcopy(VAE_CONFIG)
    config_path = VAE_DIR / "config.json"
    if config_path.exists():
        config_dict = json.loads(config_path.read_text(encoding="utf-8"))

    state_dict = load_file(ckpt_path, device="cpu")
    state_dict = convert_ldm_vae_checkpoint(state_dict, config_dict)
    model = AutoencoderKL.from_config(config_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys in VAE checkpoint")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys in VAE checkpoint")

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def _remap_comfy_to_model_keys(state_dict: dict) -> dict:
    """
    Remap ComfyUI-format checkpoint keys → model parameter names.

    z_image_convert_original_to_comfy.py applies these transforms when building
    the ComfyUI checkpoint.  We invert them here:
      x_embedder.*            → all_x_embedder.2-1.*
      final_layer.*           → all_final_layer.2-1.*
      .attention.qkv.weight   → split into to_q / to_k / to_v weights
      .attention.out.weight   → .attention.to_out.0.weight
      .attention.q_norm.weight→ .attention.norm_q.weight
      .attention.k_norm.weight→ .attention.norm_k.weight
      (to_out.0.bias was dropped by the converter — nothing to restore)
    """
    out: dict = {}
    for k, v in state_dict.items():
        # Fused QKV → separate q / k / v
        if k.endswith(".attention.qkv.weight"):
            prefix = k[: -len(".attention.qkv.weight")]
            q, k_w, v_w = v.chunk(3, dim=0)
            out[prefix + ".attention.to_q.weight"] = q
            out[prefix + ".attention.to_k.weight"] = k_w
            out[prefix + ".attention.to_v.weight"] = v_w
            continue

        k_out = k
        # Top-level embedder / final-layer renames
        if k_out.startswith("x_embedder."):
            k_out = "all_x_embedder.2-1." + k_out[len("x_embedder."):]
        elif k_out.startswith("final_layer."):
            k_out = "all_final_layer.2-1." + k_out[len("final_layer."):]

        # Attention sub-module renames
        k_out = k_out.replace(".attention.out.weight",    ".attention.to_out.0.weight")
        k_out = k_out.replace(".attention.q_norm.weight", ".attention.norm_q.weight")
        k_out = k_out.replace(".attention.k_norm.weight", ".attention.norm_k.weight")

        out[k_out] = v
    return out


def _load_transformer(device: str, block_offload: bool = False) -> ZImageTransformer2DModel:
    """Load the ComfyUI single-file checkpoint from local disk into our nn module."""
    ckpt_path = _find_local_safetensors(DIFFUSION_MODELS_DIR)
    print(f"  Transformer checkpoint: {ckpt_path.name}")

    state_dict = load_file(ckpt_path, device="cpu")

    # Strip common outer wrapper prefixes if present
    for prefix in ("model.", "transformer."):
        if all(k.startswith(prefix) for k in state_dict):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            break

    # Remap ComfyUI key names → model parameter names
    state_dict = _remap_comfy_to_model_keys(state_dict)

    model = ZImageTransformer2DModel(**TRANSFORMER_CONFIG)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys in transformer checkpoint")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys in transformer checkpoint")

    load_device = "cpu" if block_offload else device
    model = model.to(device=load_device, dtype=torch.bfloat16)
    if block_offload:
        model.enable_block_offload(device)
    else:
        model.disable_block_offload()
    model.eval()
    return model


# ── Timestep / shift helpers (ported from diffusers pipeline_z_image.py) ──────

def _calculate_shift(
    image_seq_len: int,
    base_image_seq_len: int = 256,
    max_image_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    b = base_shift - m * base_image_seq_len
    return image_seq_len * m + b


# ── Text encoding ──────────────────────────────────────────────────────────────

def _encode_prompt(
    tokenizer: PreTrainedTokenizerBase,
    text_encoder: PreTrainedModel,
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
            use_cache=False,
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
    scheduler: FlowMatchEulerDiscreteScheduler
    device: str
    transformer: ZImageTransformer2DModel | None = None
    transformer_resident_on_device: bool = False


def _clear_device_cache(device: str) -> None:
    if device == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()
    elif device == "cuda" and hasattr(torch, "cuda"):
        torch.cuda.empty_cache()


def _unload_stage_model(model: torch.nn.Module, device: str) -> None:
    # NOTE: caller must not hold any other reference to `model` when calling
    # this, otherwise `del model` here only removes the local copy and gc.collect
    # cannot reclaim the object. Drop the caller's variable first, then call.
    del model
    gc.collect()
    _clear_device_cache(device)


def _encode_prompts_staged(
    components: PipelineComponents,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
    print("Loading tokenizer for prompt encoding…")
    tokenizer = _load_tokenizer()
    text_encoder: torch.nn.Module | None = None
    active_text_device: str | None = None
    candidate_devices = [components.device]
    if components.device != "cpu":
        candidate_devices.append("cpu")

    try:
        for text_device in candidate_devices:
            print(f"Using {text_device} for prompt encoding stage")
            try:
                print("Loading text encoder for prompt encoding…")
                text_encoder = _load_text_encoder(text_device)
                active_text_device = text_device

                pos_embeds = _encode_prompt(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    prompt=prompt,
                    device=text_device,
                )

                do_cfg = guidance_scale > 0.0
                neg_embeds = None
                if do_cfg:
                    neg_prompt = negative_prompt if negative_prompt else ""
                    neg_embeds = _encode_prompt(
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        prompt=neg_prompt,
                        device=text_device,
                    )

                if text_device != components.device:
                    pos_embeds = pos_embeds.to(device=components.device, dtype=torch.bfloat16)
                    if neg_embeds is not None:
                        neg_embeds = neg_embeds.to(device=components.device, dtype=torch.bfloat16)

                return pos_embeds, neg_embeds, do_cfg
            except Exception as exc:
                print(f"  [warn] Prompt encoding on {text_device} failed: {exc}")
                if text_encoder is not None:
                    print("Unloading text encoder…")
                    _te_device = text_device
                    del text_encoder
                    text_encoder = None
                    active_text_device = None
                    gc.collect()
                    _clear_device_cache(_te_device)

                if text_device == candidate_devices[-1]:
                    raise

                print("  [warn] Falling back to CPU prompt encoding")

        raise RuntimeError("Prompt encoding failed on all candidate devices")
    finally:
        print("Unloading tokenizer…")
        del tokenizer
        if text_encoder is not None:
            print("Unloading text encoder…")
            te_device = active_text_device or candidate_devices[0]
            # Drop the only remaining caller-side reference BEFORE gc.collect so
            # the model object is actually freed from VRAM.
            del text_encoder
            gc.collect()
            _clear_device_cache(te_device)


def load_pipeline() -> tuple[PipelineComponents, str]:
    """Load lightweight runtime components; models/tokenizer are loaded per request."""
    device = _get_device()
    print(f"Device: {device}")

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    print("Preloading transformer to CPU…")
    transformer = _load_transformer(device="cpu", block_offload=False)

    components = PipelineComponents(
        scheduler=scheduler,
        device=device,
        transformer=transformer,
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
    attn_chunk_size: int | None = None,
    attn_backend: str = "chunked",
    generation_mode: str = "offload",
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
        attn_chunk_size: Query chunk size for chunked transformer attention.
        attn_backend:    Attention backend name ("chunked" or "native").
        generation_mode: Transformer runtime mode:
                 - "offload": keep master weights on CPU and offload blocks.
                 - "persistent": keep full transformer resident on device.
    """
    device = components.device

    if attn_chunk_size is not None and attn_chunk_size <= 0:
        raise ValueError(f"attn_chunk_size must be > 0, got {attn_chunk_size}")

    if attn_backend not in {"chunked", "native"}:
        raise ValueError(f"attn_backend must be 'chunked' or 'native', got {attn_backend!r}")

    if generation_mode not in {"offload", "persistent"}:
        raise ValueError(f"generation_mode must be 'offload' or 'persistent', got {generation_mode!r}")

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

    # ── Encode text (load/unload text encoder on-demand)
    pos_embeds, neg_embeds, do_cfg = _encode_prompts_staged(
        components=components,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
    )

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

    # ── Denoising loop (use preloaded transformer)
    use_offload = generation_mode == "offload"

    if use_offload and device == "cpu":
        print("  [warn] offload mode has no effect on CPU; using CPU execution")
        use_offload = False

    if use_offload:
        if components.transformer_resident_on_device and device != "cpu":
            print("Switching transformer from persistent GPU mode back to CPU for offload mode…")
            components.transformer.to(device="cpu", dtype=torch.bfloat16)
            components.transformer_resident_on_device = False
            _clear_device_cache(device)
        print("Transformer denoising with block offload mode…")
        components.transformer.enable_block_offload(device)
    else:
        components.transformer.disable_block_offload()
        if device != "cpu" and not components.transformer_resident_on_device:
            print("Loading transformer to GPU in persistent mode…")
            components.transformer.to(device=device, dtype=torch.bfloat16)
            components.transformer_resident_on_device = True
        elif device != "cpu":
            print("Using resident transformer on GPU (persistent mode)…")

    transformer = components.transformer

    backend_ctx = attention_backend(attn_backend)
    chunk_ctx = attention_chunk_size(attn_chunk_size) if attn_chunk_size is not None else contextlib.nullcontext()
    try:
        with backend_ctx, chunk_ctx:
            for t in tqdm(timesteps, total=len(timesteps), desc="Denoising", leave=True):
                timestep = t.expand(latents.shape[0])
                t_norm = (1000 - timestep) / 1000      # normalise to [0, 1]

                if do_cfg:
                    # Duplicate for positive + negative
                    lat_in = latents.to(transformer.dtype)
                    lat_in = lat_in.unsqueeze(2)       # [1, C, 1, H', W']
                    lat_list = list(lat_in.repeat(2, 1, 1, 1, 1).unbind(dim=0))
                    cap_list = [pos_embeds, neg_embeds]
                    t_in = t_norm.repeat(2)
                else:
                    lat_in = latents.to(transformer.dtype).unsqueeze(2)
                    lat_list = list(lat_in.unbind(dim=0))
                    cap_list = [pos_embeds]
                    t_in = t_norm

                model_out = transformer(
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
    finally:
        if use_offload:
            components.transformer.disable_block_offload()
            print("Transformer block offload complete, keeping on CPU for next request")
        elif device == "cpu":
            components.transformer_resident_on_device = False

    # ── VAE decode (load/unload VAE on-demand)
    print("Loading VAE for decode…")
    vae = _load_vae(device)
    try:
        latents = latents.to(vae.dtype)
        scaling = vae.config.scaling_factor
        shift   = getattr(vae.config, "shift_factor", 0.0)
        latents = (latents / scaling) + shift

        image_tensor = vae.decode(latents, return_dict=False)[0]
    finally:
        print("Unloading VAE…")
        _unload_stage_model(vae, device)

    # image_tensor: [1, 3, H, W] in [-1, 1] (standard VAE output)
    image_tensor = (image_tensor.float().clamp(-1, 1) + 1) / 2   # → [0, 1]
    image_np = (image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    image = Image.fromarray(image_np)

    return image
