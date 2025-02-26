import gradio as gr

from llavamed_inference_openvino import run_inference_image


def process_inputs(image, question):
    if image is None:
        return "Please upload an image."
    if not question:
        return "Please enter a question."
    return run_inference_image(image, question)


def reset_inputs():
    return None, "", ""


with gr.Blocks() as demo:
    gr.Markdown("## LLaVA-Med 1.5 OpenVINO Demo")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload an Image")
        text_input = gr.Textbox(label="Enter a Question")

    output_text = gr.Textbox(label="Output", interactive=False)

    with gr.Row():
        process_button = gr.Button("Process")
        reset_button = gr.Button("Reset")
    process_button.click(process_inputs, inputs=[image_input, text_input], outputs=output_text)
    reset_button.click(reset_inputs, inputs=[], outputs=[image_input, text_input, output_text])


demo.launch(server_port=7788)
