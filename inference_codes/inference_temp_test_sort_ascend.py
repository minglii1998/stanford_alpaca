import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch
import transformers
import argparse
from transformers import (
    LlamaForCausalLM, LlamaTokenizer, GenerationConfig)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_type', default="llama", choices=['llama', 'chatglm', 'bloom'])
parser.add_argument('--trained_check_path', default="out_try", type=str)
args = parser.parse_args()


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

if device == "cuda":
    if args.model_type == "llama":
        model = LlamaForCausalLM.from_pretrained(
            args.trained_check_path,
            device_map="auto",
            cache_dir="../cache/"
        )
        
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            'decapoda-research/llama-7b-hf',
            cache_dir="../cache/",
            model_max_length=512, # 512 by default in alpaca offical training
            padding_side="right",
            use_fast=False,
        )

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


model.eval()


def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=1,
    top_k=40,
    num_beams=4,
    max_new_tokens=512,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()

def evaluate_raw(
    instruction,
    input=None,
    max_new_tokens=512,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    # prompt = instruction
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )
    output = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]

    return output


if __name__ == "__main__":
    import json
    import re

    file_path = 'data/generated_test_data/generated_sort_ascend_data_200.json'
    print(file_path)
    with open(file_path, "r") as f:
        data = json.load(f)

    correct_count = 0
    for i,sample_i in enumerate(data):
        print('==================/n',i,'/',len(data))

        instruction = sample_i['instruction']
        input_ = sample_i['input']
        response = evaluate_raw(instruction,input_)
        response = response.split('### Response:')[1].strip()

        print('Instruction:',sample_i['instruction'])
        print('Input:',sample_i['input'])
        print("GT:")
        print(sample_i['output'])
        print("Response:")
        print(response)

        output_i = response.lower()
        numbers_response = re.findall(r'-?\d+\.?\d*',output_i)

        output_i = sample_i['output'].lower()
        numbers_gt = re.findall(r'-?\d+\.?\d*',output_i)

        if len(numbers_response) == 0:
            print("False")
        elif len(numbers_response) != len(numbers_gt):
            print("False")
        else:
            if numbers_response == numbers_gt:
                correct_count += 1
                print("True")
            else:
                print("False")

    print('Final Acc:', correct_count/len(data))
