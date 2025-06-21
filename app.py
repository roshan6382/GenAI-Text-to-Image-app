import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
import uuid
import os

# Load model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to("cpu")

# Output folder
os.makedirs("output", exist_ok=True)

style_dict = {
    "None": "",
    "Photorealistic": ", ultra realistic, 8k, cinematic lighting",
    "Anime": ", anime style, bright colors, cel shading",
    "Digital Painting": ", digital art, brush strokes, rich texture",
    "Pencil Sketch": ", pencil sketch, black and white, rough lines",
    "Cyberpunk": ", cyberpunk cityscape, neon glow, futuristic tech",
    "Fantasy": ", epic fantasy, mystical lighting, dragons, magic"
}

generated_images = []

def generate(prompt, style):
    style_suffix = style_dict.get(style, "")
    full_prompt = prompt + style_suffix
    image = pipe(full_prompt).images[0]

    filename = f"output/{uuid.uuid4().hex}.png"
    image.save(filename)

    # Track latest 4 images
    generated_images.append(filename)
    if len(generated_images) > 4:
        generated_images.pop(0)

    return image, filename, [gr.Image(value=img_path) for img_path in reversed(generated_images)]

with gr.Blocks(theme=gr.themes.Soft(primary_hue="violet", secondary_hue="purple")) as demo:
    gr.Markdown("""
    <h1 style='text-align: center;'>🎨 GenAI Text-to-Image App</h1>
    <p style='text-align: center; font-size: 18px;'>Describe anything with words — and let AI turn it into art.</p>
    <p style='text-align: center;'>Built with 🤗 Diffusers + ❤️ Gradio | 100% Free & Local</p>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="📝 Enter your prompt", placeholder="e.g., a surreal dreamscape with floating clocks", lines=2)
            style = gr.Dropdown(label="🎨 Choose a Style", choices=list(style_dict.keys()), value="None")

            gr.Markdown("**✨ Example Prompts:**")
            with gr.Row():
                gr.Button("🏙️ Cyberpunk street").click(fn=lambda: "a futuristic cyberpunk street at night", outputs=prompt)
                gr.Button("🐉 Fantasy dragon").click(fn=lambda: "a dragon flying over ancient ruins", outputs=prompt)
                gr.Button("🎠 Dreamy carnival").click(fn=lambda: "a magical carnival under the stars", outputs=prompt)

            submit = gr.Button("🚀 Generate Image", variant="primary")

        with gr.Column(scale=1):
            output_img = gr.Image(label="🎨 AI Output", type="pil", show_label=True)
            download_link = gr.File(label="⬇️ Download Image", visible=False)

    gallery = gr.Gallery(label="🖼️ Your Last 4 Generations", columns=2, height="auto")

    submit.click(fn=generate, inputs=[prompt, style], outputs=[output_img, download_link, gallery])

demo.launch()
