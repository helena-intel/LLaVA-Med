import gradio as gr

from llavamed_inference_openvino import run_inference_image

css = """
.text textarea {font-size: 24px !important;}
"""

def process_inputs(image, question):
    if image is None:
        return "Please upload an image."
    if not question:
        return "Please enter a question."
    return run_inference_image(image, question)


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

    gr.Markdown("NOTE: This OpenVINO model is unvalidated. Results are provisional and may contain errors. Use this demo to explore AI PC and OpenVINO optimizations")
    gr.Markdown("Source model: [microsoft/LLaVA-Med](https://github.com/microsoft/LLaVA-Med). For research purposes only.")

    process_button.click(process_inputs, inputs=[image_input, text_input], outputs=output_text)
    text_input.submit(process_inputs, inputs=[image_input, text_input], outputs=output_text)  
    reset_button.click(reset_inputs, inputs=[], outputs=[image_input, text_input, output_text])

demo.launch(server_port=7788)
