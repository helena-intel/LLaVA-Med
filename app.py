"""
Gradio app for Llava-Med inference with OpenVINO
See https://github.com/helena-intel/Llava-Med for installation instructions

Usage: python app.py [model] [image_device] [llm_device]

All arguments are optional. Default values for model, image_device and llm_device are:
model: llava-med-imf16-llmint8
image_device: NPU
llm_device: GPU
"""

import argparse
import time

import gradio as gr

from llavamed_inference_openvino import LlavaMedOV

css = """
.text textarea {font-size: 24px !important;}
"""


def process_inputs(image, question):
    if image is None:
        return "Please upload an image."
    if not question:
        return "Please enter a question."
    return llavamed.run_inference_image(image, question)


def reset_inputs():
    return None, "", ""


with gr.Blocks(css=css) as demo:
    gr.Markdown("# LLaVA-Med 1.5 OpenVINO Demo")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload an Image", height=300, width=500)
        with gr.Column():
            text_input = gr.Textbox(label="Enter a Question", elem_classes="text")
            output_text = gr.Textbox(label="Answer", interactive=False, elem_classes="text")

    with gr.Row():
        process_button = gr.Button("Process")
        reset_button = gr.Button("Reset")

    gr.Markdown(
        "NOTE: This model is for research purposes only and can make mistakes. Use this demo to explore AI PC and OpenVINO optimizations."
    )
    gr.Markdown("Source model: [microsoft/LLaVA-Med](https://github.com/microsoft/LLaVA-Med).")

    process_button.click(process_inputs, inputs=[image_input, text_input], outputs=output_text)
    text_input.submit(process_inputs, inputs=[image_input, text_input], outputs=output_text)
    reset_button.click(reset_inputs, inputs=[], outputs=[image_input, text_input, output_text])


parser = argparse.ArgumentParser()
parser.add_argument("model", nargs='?', default="llava-med-imf16-llmint8")
parser.add_argument("image_device", nargs='?', default="NPU")
parser.add_argument("llm_device", nargs='?', default="GPU")
args = parser.parse_args()

print(f"Loading {args.model} to {args.image_device} (image) and {args.llm_device} (llm)")
start = time.perf_counter()
llavamed = LlavaMedOV(args.model)
llavamed.load_model(args.image_device, args.llm_device)
end = time.perf_counter()
print(f"Model loading completed in {end-start:.2f} seconds")

demo.launch(server_port=7788)
