"""
app.py – Gradio web UI for Z-Image-Turbo on Intel XPU.

Run:
    python app.py
"""

import gradio as gr
from pipeline import load_pipeline, generate

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
    transformer_block_offload: bool,
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
        transformer_block_offload=transformer_block_offload,
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

            attn_chunk_size = gr.Slider(
                label="Attention chunk size (lower = less VRAM, slower)",
                minimum=64,
                maximum=1024,
                step=32,
                value=256,
            )

            attn_backend = gr.Radio(
                label="Attention backend",
                choices=["chunked", "native"],
                value="chunked",
            )

            transformer_block_offload = gr.Checkbox(
                label="Transformer block-by-block offload (much lower VRAM, slower)",
                value=True,
            )

            seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated image", type="pil")

    generate_btn.click(
        fn=run_inference,
        inputs=[prompt, negative_prompt, width, height, steps, guidance_scale, seed, attn_backend, attn_chunk_size, transformer_block_offload],
        outputs=[output_image],
    )

    gr.Examples(
        examples=[
            ["A futuristic cityscape at sunset, ultra detailed, cinematic lighting", "", 1024, 1024, 9, 0.0, 42, "chunked", 256, True],
            ["A close-up portrait of a red panda in a forest, studio lighting", "blurry, low quality", 1024, 1024, 9, 0.0, 7, "chunked", 256, True],
            ["An oil painting of a sailing ship in a storm", "", 1024, 1024, 9, 0.0, -1, "chunked", 256, True],
        ],
        inputs=[prompt, negative_prompt, width, height, steps, guidance_scale, seed, attn_backend, attn_chunk_size, transformer_block_offload],
        # Keep examples as input presets only. Triggering generation directly from
        # examples can create long-running startup/background UI states.
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
