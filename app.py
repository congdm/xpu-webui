"""
app.py – Gradio web UI for Z-Image-Turbo on Intel XPU.

Run:
    python app.py
"""

import gradio as gr
import json
import os
from pathlib import Path
from pipeline import load_pipeline, generate

# ---------------------------------------------------------------------------
# Configuration handling
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "webui_config.json"

def load_webui_config() -> dict:
    """Load web UI configuration from JSON file, return defaults if missing."""
    default_config = {
        "attn_chunk_size": 256,
        "attn_backend": "chunked",
        "generation_mode": "offload"
    }
    
    if not CONFIG_PATH.exists():
        print(f"Config file not found at {CONFIG_PATH}, using defaults")
        return default_config
    
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Ensure all required keys exist
        for key in default_config:
            if key not in config:
                print(f"Config missing key '{key}', using default")
                config[key] = default_config[key]
        
        print(f"Loaded config from {CONFIG_PATH}")
        return config
    except Exception as e:
        print(f"Error loading config from {CONFIG_PATH}: {e}, using defaults")
        return default_config

def save_webui_config(config: dict) -> bool:
    """Save web UI configuration to JSON file."""
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Saved config to {CONFIG_PATH}")
        return True
    except Exception as e:
        print(f"Error saving config to {CONFIG_PATH}: {e}")
        return False

def save_current_config(attn_chunk_size: int, attn_backend: str, generation_mode: str) -> None:
    """Save current settings values to config file."""
    config = {
        "attn_chunk_size": attn_chunk_size,
        "attn_backend": attn_backend,
        "generation_mode": generation_mode
    }
    success = save_webui_config(config)
    if success:
        gr.Info("✅ Config saved successfully!")
    else:
        gr.Warning("❌ Failed to save config.")

# Load configuration at startup
webui_config = load_webui_config()

# ---------------------------------------------------------------------------
# Initialize lightweight pipeline at startup
# ---------------------------------------------------------------------------
print("Initializing Z-Image-Turbo pipeline…")
pipe, device = load_pipeline()
print(f"Pipeline initialized on device: {device} (tokenizer and models are loaded per request)")

# ---------------------------------------------------------------------------
# Inference callback
# ---------------------------------------------------------------------------

def run_inference(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    attn_backend: str,
    attn_chunk_size: int,
    generation_mode: str,
):
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    image = generate(
        components=pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        attn_backend=attn_backend,
        attn_chunk_size=attn_chunk_size,
        generation_mode=generation_mode,
    )

    return image

# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Z-Image-Turbo · Intel XPU", analytics_enabled=False) as demo:
    gr.Markdown(
        """
        # 🖼️ Z-Image-Turbo · Intel XPU
        **Model:** [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo)  
        **Device:** """ + device + """
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("Generation"):
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="A photo of an astronaut riding a horse on Mars",
                        lines=3,
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative prompt",
                        placeholder="blurry, low quality, watermark",
                        lines=2,
                    )

                    with gr.Row():
                        width = gr.Slider(
                            label="Width", minimum=512, maximum=1536, step=16, value=1024
                        )
                        height = gr.Slider(
                            label="Height", minimum=512, maximum=1536, step=16, value=1024
                        )

                    with gr.Row():
                        steps = gr.Slider(
                            label="Inference steps", minimum=1, maximum=30, step=1, value=9
                        )
                        guidance_scale = gr.Slider(
                            label="Guidance scale (0 = turbo/no CFG)", minimum=0.0, maximum=10.0, step=0.5, value=0.0
                        )

                    seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

                    generate_btn = gr.Button("Generate", variant="primary")
                
                with gr.Tab("Settings"):
                    attn_chunk_size = gr.Slider(
                        label="Attention chunk size (lower = less VRAM, slower)",
                        minimum=64,
                        maximum=1024,
                        step=32,
                        value=webui_config["attn_chunk_size"],
                    )

                    attn_backend = gr.Radio(
                        label="Attention backend",
                        choices=["chunked", "native"],
                        value=webui_config["attn_backend"],
                    )

                    generation_mode = gr.Radio(
                        label="Transformer runtime mode",
                        choices=[
                            ("Offload (default, lower VRAM)", "offload"),
                            ("Persistent on GPU (faster, high VRAM)", "persistent"),
                        ],
                        value=webui_config["generation_mode"],
                    )

                    save_config_btn = gr.Button("Save config", variant="secondary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated image", type="pil")

    generate_btn.click(
        fn=run_inference,
        inputs=[prompt, negative_prompt, width, height, steps, guidance_scale, seed, attn_backend, attn_chunk_size, generation_mode],
        outputs=[output_image],
    )

    save_config_btn.click(
        fn=save_current_config,
        inputs=[attn_chunk_size, attn_backend, generation_mode],
    )

    gr.Examples(
        examples=[
            ["A futuristic cityscape at sunset, ultra detailed, cinematic lighting", "", 1024, 1024, 9, 0.0, 42],
            ["A close-up portrait of a red panda in a forest, studio lighting", "blurry, low quality", 1024, 1024, 9, 0.0, 7],
            ["An oil painting of a sailing ship in a storm", "", 1024, 1024, 9, 0.0, -1],
        ],
        # Examples should only change normal generation settings, not runtime/offload tuning.
        inputs=[prompt, negative_prompt, width, height, steps, guidance_scale, seed],
        # Keep examples as input presets only. Triggering generation directly from
        # examples can create long-running startup/background UI states.
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
