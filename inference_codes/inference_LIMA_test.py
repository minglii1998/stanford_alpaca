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

    import time
    import json
    file_lima = 'data/lima_datas/lima_test_data.json'
    with open(file_lima, "r") as f:
        data_lima = json.load(f)
    
    trained_check_path = args.trained_check_path
    model_name = trained_check_path.split('/')[-2]
    check_point_name = trained_check_path.split('/')[-1]

    save_name = 'lima_test_set'
    save_dir = os.path.join('logs',save_name,model_name,check_point_name)
    os.makedirs(save_dir,exist_ok=True)
    save_file = os.path.join(save_dir,'result.json')

    start_time = time.time()
    new_data = []
    for i,data_i in enumerate(data_lima):
        instruction_i = data_i['instruction']

        response = evaluate_raw(instruction_i)
        response = response.split('### Response:')[-1]

        new_sample = {}
        new_sample['instruction'] = instruction_i
        new_sample['response'] = response
        new_data.append(new_sample)

        print("Response:")
        print(response)
        print()

        with open('temp_see.txt','a') as f:
            f.write('==========\n')
            f.write(instruction_i+'\n')
            f.write('====\n')
            f.write(response+'\n')
            f.write('==========\n')

    print('Time used:',(time.time()-start_time)/60,(min))
    print('New data len \n',len(new_data))
    with open(save_file, "w") as fw:
        json.dump(new_data, fw, indent=4)