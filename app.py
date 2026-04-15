"""
app.py – Gradio web UI for Z-Image-Turbo on Intel XPU.

Run:
    python app.py
"""

import gradio as gr
from pipeline import load_pipeline, generate

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
print("Loading Z-Image-Turbo pipeline…")
pipe, device = load_pipeline()
print(f"Pipeline loaded on device: {device}")

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
):
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    image = generate(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    return image

# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Z-Image-Turbo · Intel XPU") as demo:
    gr.Markdown(
        """
        # 🖼️ Z-Image-Turbo · Intel XPU
        **Model:** [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)  
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
                    label="Width", minimum=512, maximum=1536, step=64, value=1024
                )
                height = gr.Slider(
                    label="Height", minimum=512, maximum=1536, step=64, value=1024
                )

            with gr.Row():
                steps = gr.Slider(
                    label="Inference steps", minimum=1, maximum=20, step=1, value=4
                )
                guidance_scale = gr.Slider(
                    label="Guidance scale", minimum=0.0, maximum=10.0, step=0.5, value=0.0
                )

            seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated image", type="pil")

    generate_btn.click(
        fn=run_inference,
        inputs=[prompt, negative_prompt, width, height, steps, guidance_scale, seed],
        outputs=output_image,
    )

    gr.Examples(
        examples=[
            ["A futuristic cityscape at sunset, ultra detailed, cinematic lighting", "", 1024, 1024, 4, 0.0, 42],
            ["A close-up portrait of a red panda in a forest, studio lighting", "blurry, low quality", 1024, 1024, 4, 0.0, 7],
            ["An oil painting of a sailing ship in a storm", "", 1024, 1024, 4, 0.0, -1],
        ],
        inputs=[prompt, negative_prompt, width, height, steps, guidance_scale, seed],
        outputs=output_image,
        fn=run_inference,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
