import os
import json
import argparse
from copy import deepcopy
import itertools
from typing import Any
from operator import add
from pprint import pprint
from typing import List
from pathlib import Path
from tqdm import tqdm

from optimum.intel import OVModelForCausalLM
from transformers import pipeline, AutoTokenizer

import llm
import util


INSTRUCT_PROMPT = """We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with caption describing the same image.
  Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
  Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""
ROLE = 'Assistant'

# Generate instruction for GPT-4 to score the two answers.
def conv_to_str(fig_label, fig_caption, fig_context, question, ans1, ans2):
  return (f'[Context]\n'
          f'Figure Caption:\n{fig_label}: {fig_caption}\n\n'
          f'Figure Context:\n\t- {fig_context}\n\n'
          f'[Question]\n{question}\n\n'
          f'[{ROLE} 1]\n{ans1}\n\n[End of {ROLE} 1]\n\n'
          f'[{ROLE} 2]\n{ans2}\n\n[End of {ROLE} 2]\n\n'
          f'[System]\n{INSTRUCT_PROMPT}\n\n')

def compare_messages_gen(fig_label, fig_caption, fig_context, question, ans1, ans2):
  messages = [
  {"role": "system", "content": """'You are a helpful and precise assistant for checking the quality of the answer."""},
  ]
  messages.append({"role": "user", "content": conv_to_str(fig_label, fig_caption, fig_context, question, ans1, ans2)})
  return messages


def sum_list_list(x):
  return sum(item for inner_list in x for item in inner_list)

def chunk(lst, n):
  for i in range(0, len(lst), n):
    if i+(1.5*n)<len(lst):
      end = i + n
    else:
      end = len(lst)
    yield lst[i:end]
    if end==len(lst):
      return


def infer(samples, pipe=None):
    model_inst = None
    if pipe is None:
        model_inst = llm.GPT("gpt-4")
        model_name = "gpt"
    else:
        model_name = "llama"


    BATCH_SIZE = 1
    batch_samples = []
    results = []
    batch = []

    print('Starting Multimodal Chat GPT Scoring Eval')

    for sample in tqdm(samples):
        sample_copy = deepcopy(sample)
        input_msg = compare_messages_gen(sample_copy['fig_label'], sample_copy['fig_caption'], sample_copy['in_text_mention'], sample_copy['question'], sample_copy['ans1'], sample_copy['ans2'])
        batch.append(input_msg)
        batch_samples.append(sample_copy)
        if len(batch)>=BATCH_SIZE:
            if pipe is None:
                # GPT 4 API inference
                inference_results = [x.strip() for chunk_messages in chunk([x for x in batch if x], BATCH_SIZE) for x in model_inst.infer(chunk_messages)]
            else:
                # inference_results = [pipe(batch)[0][0]["generated_text"][-1]["content"]]
                filtered_batch = [x for x in batch if x]
                chunked_batches = chunk(filtered_batch, BATCH_SIZE)
                inference_results = []

                for chunk_messages in chunked_batches:
                  inferred_messages = pipe(chunk_messages)[0][0]["generated_text"][-1]["content"]
                          
                  inference_results.extend([inferred_messages])

            for item, inference_result in zip(batch_samples, inference_results):
                item[f'gpt_eval'] = inference_result
            results.extend(batch_samples)
            batch = []
            batch_samples = []
    for item, inference_result in zip(batch_samples, inference_results):
        item['gpt_eval'] = inference_result
    results.extend(batch_samples)
    print(f"Result Size: {len(results)}")
    return results


def main(args):
    answer_data = util.load_file_jsonl(args.answers_file)[:args.limit]
    question_data = util.load_file_jsonl(args.question_file)[:args.limit]

    samples = []
    for question, answer in zip(question_data, answer_data):
        question_copy = deepcopy(question)
        question['question'] = question_copy['text']
        question['ans1'] = question_copy.pop('gpt4_answer')
        question['ans2'] = answer['text']
        samples.append(question)


    pipe = None
    if args.local_model is not None:
        model = OVModelForCausalLM.from_pretrained(args.local_model)
        tokenizer = AutoTokenizer.from_pretrained(args.local_model)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    results = infer(samples, pipe=pipe)

    # Create parent directory of output score files if it doesn't exist
    os.makedirs(Path(args.scores_file).parent, exist_ok=True)

    with open(args.scores_file, 'w') as f:
       for row in results:
          f.write(json.dumps(row)+'\n')


if __name__ == '__main__':
   parser = argparse.ArgumentParser("GPT-4 Multimodal Chat Scoring", add_help=True)
   parser.add_argument("--answers-file", default="/home/sdp/helena/LLaVA-Med/answers_all/answers_ov_bf16_imfp32_llmint8.jsonl", metavar="FILE", help="path to model answer file")
   parser.add_argument("--question-file", default="/home/sdp/helena/LLaVA-Med/data/eval/llava_med_eval_qa50_qa.jsonl", metavar="FILE", help="path to multichat questions file")
   parser.add_argument("--scores-file", default="/home/sdp/helena/LLaVA-Med/scores_test.jsonl", metavar="FILE", help="path to save gpt-4 score file")
   parser.add_argument("--local-model", default="/home/sdp/helena/LLaVA-Med/Meta-LLama-3-8b-Instruct-ov", help="Local OpenVINO model to use for inference instead of GPT-4")
   parser.add_argument("--limit", type=int, default=5)

   args = parser.parse_args()
   main(args)
