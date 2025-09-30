import csv
import importlib.metadata
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import openvino as ov
import pandas as pd
from PIL import Image
from transformers import logging

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (get_model_name_from_path, process_images,
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model

logging.set_verbosity_error()

ENABLE_PERFORMANCE_METRICS = True


def perf_metrics(num_tokens, duration):
    tps = round(num_tokens / duration, 2)
    latency = round((duration / num_tokens) * 1000, 2)
    return tps, latency


class LlavaMedOV:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_name = get_model_name_from_path(self.model_path)
        self.image_device = None
        self.llm_device = None
        self.tokenizer = None
        self.model = None
        self.image_processor = None

    def load_model(self, image_device, llm_device):
        self.image_device = image_device
        self.llm_device = llm_device
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
            model_path=self.model_path,
            model_base=None,
            model_name=self.model_name,
            device=llm_device,
            openvino=True,
            image_device=image_device,
        )

    def prepare_inputs_image(self, question, image):
        conv_mode = "vicuna_v1"  # default
        qs = question.replace(DEFAULT_IMAGE_TOKEN, "").strip()
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs  # model.config.mm_use_im_start_end is False

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

        # image = Image.open(image_file)
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        return input_ids, image_tensor

    def run_inference_image(self, image, question):
        input_ids, image_tensor = self.prepare_inputs_image(question, image)
        start = time.perf_counter()
        ov_output_ids = self.model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half(),
            do_sample=False,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True,
        )
        end = time.perf_counter()
        duration = end - start
        print(f"Inference duration: {duration:.2f} seconds")
        input_length = input_ids.shape[-1]
        ov_output_ids = ov_output_ids[:, input_length:]
        output_num_tokens = ov_output_ids.shape[-1]
        answer = self.tokenizer.batch_decode(ov_output_ids, skip_special_tokens=True)[0].strip()
        print(f"Answer: {answer}")

        if ENABLE_PERFORMANCE_METRICS:
            try:
                first_latency = self.model.llm_latencies[0]
                second_latency = np.mean(self.model.llm_latencies[1:])
                avg_latency = np.mean(self.model.llm_latencies)
                image_latency = self.model.image_latencies[0]
                logfile = "llavamed_performance_debug.csv"
                tps, latency = perf_metrics(output_num_tokens, duration)
                system = ov.Core().get_property("CPU", "FULL_DEVICE_NAME")
                ov_version = importlib.metadata.version("openvino")
                perf_record = {
                    "model path": Path(self.model_path).name,
                    "system": system,
                    "openvino": ov_version,
                    "image device": self.image_device,
                    "llm device": self.llm_device,
                    "question": question,
                    "answer": answer,
                    "output tokens": output_num_tokens,
                    "duration": duration,
                    "throughput (tok/sec)": tps,
                    "llm avg latency (ms)": avg_latency,
                    "llm 1st latency (ms)": first_latency,
                    "llm 2nd latency (ms)": second_latency,
                    "image latency (ms)": image_latency,
                }

                writeheader = not Path(logfile).is_file()
                with open(logfile, "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=perf_record.keys())
                    if writeheader:
                        writer.writeheader()
                    writer.writerow(perf_record)

                min_columns = [item for item in perf_record if item not in ("question", "answer")]
                df = pd.read_csv(logfile, encoding="utf-8", encoding_errors="replace", usecols=min_columns)
                pivot = df.pivot_table(
                    index=["model path", "image device", "llm device"],
                    values=[
                        "duration",
                        "throughput (tok/sec)",
                        "llm avg latency (ms)",
                        "llm 1st latency (ms)",
                        "llm 2nd latency (ms)",
                        "image latency (ms)",
                    ],
                )
                performance_summary = pivot.T.to_markdown()
                print(performance_summary)
                with open("llavamed_performance_summary.txt", "w") as f:
                    f.write(performance_summary)
                df.to_csv("llavamed_performance.csv", encoding="utf-8")
            except Exception:
                print("Error while logging performance data:")
                traceback.print_exc()
                pass  # Never crash on writing performance metrics for any reason

        return answer


# suggested indices are indices where model output is similar to source model output
# it may still be incorrect!
# int8 image model: 2, 13, 14, 16, 17 (shorter) and  4, 5, 8 (longer)
# f32 image model : 0, 2, 7, 9, 13, 14, 15, 16, 17, 18, 19 (shorter) and 3,5,6,8 (longer)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to image")
    parser.add_argument(
        "--model", required=False, default="llava-med-imf16-llmint8", help="path to llava-med OpenVINO model directory"
    )
    parser.add_argument("--image_device", required=False, default="NPU", help="Device for image model")
    parser.add_argument("--llm_device", required=False, default="GPU", help="Device for LLM")
    args = parser.parse_args()

    print(f"Loading {args.model} to {args.image_device} (image) and {args.llm_device} (llm)")
    start = time.perf_counter()
    llavamed = LlavaMedOV(args.model)
    llavamed.load_model(args.image_device, args.llm_device)
    end = time.perf_counter()
    print(f"Model loading completed in {end-start:.2f} seconds")

    image = Image.open(args.image)
    llavamed.run_inference_image(image, "what is in this image? please elaborate")
