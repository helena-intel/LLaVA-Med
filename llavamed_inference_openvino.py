#!/usr/bin/env python
# coding: utf-8

# # LLaVA-Med 1.5 OpenVINO demo


import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image
from transformers import logging

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

logging.set_verbosity_error()
image_folder = "data\\qa50_images"


# ## Load Model and Data

model_path = "llava-med-imf16-llmint8"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path, model_base=None, model_name=model_name, device="gpu", openvino=True, image_device="gpu"
)
print("loaded models")

# ## Functions

def prepare_inputs_image(question, image):
    conv_mode = "vicuna_v1"  # default
    qs = question.replace(DEFAULT_IMAGE_TOKEN, "").strip()
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs  # model.config.mm_use_im_start_end is False

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

    # image = Image.open(image_file)
    image_tensor = process_images([image], image_processor, model.config)[0]
    return input_ids, image_tensor


def run_inference_image(image, question):
    # image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    
    # cv2.imshow("Image", image)
    # question = input("Question:\n")
    #image = Image.open(image_file)
    input_ids, image_tensor = prepare_inputs_image(question, image)

    ov_output_ids = model.generate(
        input_ids,
        images=image_tensor.unsqueeze(0).half(),
        do_sample=False,
        # no_repeat_ngram_size=3,
        max_new_tokens=1024,
        use_cache=True,
    )

    input_length = input_ids.shape[-1]
    ov_output_ids = ov_output_ids[:, input_length:]
    answer = tokenizer.batch_decode(ov_output_ids, skip_special_tokens=True)[0].strip()
    # print(f"Answer: {answer}")
    return answer

# suggested indices are indices where model output is similar to source model output
# it may still be incorrect!
# int8 image model: 2, 13, 14, 16, 17 (shorter) and  4, 5, 8 (longer)
# f32 image model : 0, 2, 7, 9, 13, 14, 15, 16, 17, 18, 19 (shorter) and 3,5,6,8 (longer)

if __name__ == "__main__":
   import sys
   image_file = sys.argv[1]
   run_inference_image(model, image_file)


