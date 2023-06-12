import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import sys
import torch
import transformers
import argparse
from transformers import (
    LlamaForCausalLM, LlamaTokenizer, GenerationConfig)

import re
import json
import random
from statistics import mean

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--inference_data_path', default="data/data_cot_test", type=str)
parser.add_argument('--model_type', default="llama", choices=['llama'])
parser.add_argument('--top_p', type=float, default=1, help='top_p')
parser.add_argument('--trained_check_path', default="out_try", type=str)
parser.add_argument("--dataset", type=str, default="addsub", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "bigbench_date", "object_tracking", "coin_flip", "last_letters"], help="dataset used for experiment")
parser.add_argument("--prompt_type", type=int, default=8, help="A trigger sentence that elicits a model to execute chain of thought")
parser.add_argument('--record_save_dir', default="record_test", type=str)
parser.add_argument('--record_special_tag', default="", type=str)
parser.add_argument('--instance_percent', type=float, default=1, help='how much percent of instance to be tested')
parser.add_argument("--max_new_tokens", type=int, default=2048, help="whether to limit output length")
parser.add_argument('--discard_repeat', type=int, default=0, help='whether to discard the repeated sentence')

# parser.add_argument('--data', type=str, default="alpaca-cot", help='the data used for instructing tuning')
# parser.add_argument('--size', type=str, default="7", help='the size of llama model')
# parser.add_argument('--model_name_or_path', default="decapoda-research/llama-7b-hf", type=str)
args = parser.parse_args()

def get_dataset():

    def shuffleDict(d):
        keys = list(d.keys())
        random.shuffle(keys)
        [(key, d[key]) for key in keys]
        random.shuffle(keys)
        [(key, d[key]) for key in keys]
        random.shuffle(keys)
        keys = [(key, d[key]) for key in keys]
        #keys = d(keys)
        return dict(keys)

    questions = []
    answers = []
    decoder = json.JSONDecoder()
    if args.dataset == "aqua":
        dataset_path = os.path.join(args.inference_data_path,'AQuA/test.json')
        # dataset_path = "data/data_cot_test/AQuA/test.json"
        with open(dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "(" + "(".join(json_res["options"])
                choice = choice.replace("(", " (").replace(")", ") ")
                choice = "Answer Choices:" + choice
                questions.append(json_res["question"].strip() + " " + choice)
                answers.append(json_res["correct"])

    elif args.dataset == "gsm8k":
        dataset_path = os.path.join(args.inference_data_path,'grade-school-math/test.jsonl')
        # dataset_path = "data/data_cot_test/grade-school-math/test.jsonl"
        with open(dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1])

    elif args.dataset == "commonsensqa":
        dataset_path = os.path.join(args.inference_data_path,'CommonsenseQA/dev_rand_split.jsonl')
        # dataset_path = "data/data_cot_test/CommonsenseQA/dev_rand_split.jsonl"
        with open(dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                questions.append(json_res["question"]["stem"].strip() + " " + choice)
                answers.append(json_res["answerKey"])

    elif args.dataset in ("addsub", "multiarith", "singleeq"):
        if args.dataset == 'addsub':
            dataset_path = os.path.join(args.inference_data_path,'AddSub/AddSub.json')
            # dataset_path = "data/data_cot_test/AddSub/AddSub.json"
        elif args.dataset == 'multiarith':
            dataset_path = os.path.join(args.inference_data_path,'MultiArith/MultiArith.json')
            # dataset_path = "data/data_cot_test/MultiArith/MultiArith.json"
        elif args.dataset == 'singleeq':
            dataset_path = os.path.join(args.inference_data_path,'SingleEq/questions.json')
            # dataset_path = "data/data_cot_test/SingleEq/questions.json"
        with open(dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)

    elif args.dataset == "strategyqa":
        dataset_path = os.path.join(args.inference_data_path,'StrategyQA/task.json')
        # dataset_path = "data/data_cot_test/StrategyQA/task.json"
        with open(dataset_path) as f:
            json_data = json.load(f)["examples"]
            for line in json_data:
                q = line["input"].strip()
                a = int(line["target_scores"]["Yes"])
                if a == 1:
                    a = "yes"
                else:
                    a = "no"
                questions.append(q)
                answers.append(a)

    elif args.dataset == "svamp":
        dataset_path = os.path.join(args.inference_data_path,'SVAMP/SVAMP.json')
        # dataset_path = "data/data_cot_test/SVAMP/SVAMP.json"
        with open(dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)

    elif args.dataset in ("bigbench_date", "object_tracking"):
        if args.dataset == 'bigbench_date':
            dataset_path = os.path.join(args.inference_data_path,'Bigbench_Date/task.json')
            # dataset_path = "data/data_cot_test/Bigbench_Date/task.json"
        elif args.dataset == 'object_tracking':
            dataset_path = os.path.join(args.inference_data_path,'Bigbench_object_tracking/task.json')
            # dataset_path = "data/data_cot_test/Bigbench_object_tracking/task.json"
        with open(dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            if args.dataset == "bigbench_date":
                choice_index = ['A','B','C','D','E','F']
            elif args.dataset in ("object_tracking"):
                choice_index = ['A','B','C']
            else:
                raise ValueError("dataset is not properly defined ...")
            for line in json_data:
                q = line["input"].strip()
                if args.dataset == "bigbench_date":
                    choice = "Answer Choices:"
                    # Randomly shuffle the answer choice dictionary because the original answer is always A ...
                    choice_dic = shuffleDict(line["target_scores"])
                elif args.dataset == "object_tracking":
                    choice = "\nWhich choice is true ? Answer Choices:"
                    choice_dic = line["target_scores"]
                else:
                    raise ValueError("dataset is not properly defined ...")
                for i, key_value in enumerate(choice_dic.items()):
                    key, value = key_value
                    choice += " ("
                    choice += choice_index[i]
                    choice += ") "
                    choice += key
                    if value == 1:
                        a = choice_index[i]
                        #a = key
                q = q + " " + choice
                questions.append(q)
                answers.append(a)     

    elif args.dataset in ("coin_flip", "last_letters"):
        if args.dataset == 'coin_flip':
            dataset_path = os.path.join(args.inference_data_path,'coin_flip/coin_flip.json')
            # dataset_path = "data/data_cot_test/coin_flip/coin_flip.json"
        elif args.dataset == 'last_letters':
            dataset_path = os.path.join(args.inference_data_path,'last_letters/last_letters.json')
            # dataset_path = "data/data_cot_test/last_letters/last_letters.json"
        with open(dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            for line in json_data:
                q = line["question"]
                a = line["answer"]
                questions.append(q)
                answers.append(a)

    else:
        raise ValueError("dataset is not properly defined ...")
    
    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)
    
    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))
    
    return questions, answers

questions, answers = get_dataset()

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

model.eval()

def discard_repeat_s_backup(response_mid):
    new_sentence_list = []
    sentence_list = response_mid.split('\n')
    response_token_idx = 0
    for i in range(len(sentence_list)):
        new_sentence_list.append(sentence_list[i])
        if sentence_list[i] == '### Response:':
            response_token_idx = i
            break
    response_sentence_list = sentence_list[response_token_idx+1:]
    response_sentence_set = set(response_sentence_list)

    for sent in response_sentence_list:
        if sent in response_sentence_set:
            new_sentence_list.append(sent)
            response_sentence_set.remove(sent)
    
    new_response = '\n'.join(new_sentence_list)
    return new_response

def discard_repeat_s(response_mid):
    # print('**********************************************')
    # print('response_mid\n',response_mid)
    response_mid = response_mid.replace('⁇  ','')

    new_sentence_list = []
    sentence_list_ = response_mid.split('\n')
    sentence_list = [sent.strip() for sent in sentence_list_]
    response_sentence_set = set(sentence_list)

    # print('sentence_list\n',sentence_list)
    # print('response_sentence_set\n',response_sentence_set)

    for sent in sentence_list:
        if sent in response_sentence_set:
            response_sentence_set.remove(sent)
            if sent in ['### Response:','### Input:']:
                new_sentence_list.append('\n'+sent)
            else:
                new_sentence_list.append(sent)
    
    new_response = '\n'.join(new_sentence_list)

    # print('new_sentence_list\n',new_sentence_list)
    # print('new_response\n',new_response)
    # print('**********************************************')
    return new_response


def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=512,
    **kwargs,
):
    # prompt = generate_prompt(instruction, input)
    prompt = instruction
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
    delete_first_zero = s[1:]
    output = tokenizer.decode(delete_first_zero)
    # return output.split("### Response:")[1].strip()
    return output

def evaluate_raw(
    instruction,
    input=None,
    max_new_tokens=512,
    **kwargs,
):
    # prompt = generate_prompt(instruction, input)
    prompt = instruction
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )
    output = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
    
    # s = generation_output.sequences[0]
    # delete_first_zero = s[1:]
    # output = tokenizer.decode(delete_first_zero)

    return output

def get_prompt_simple(x,y,type_flag=0,with_q=True,need_x=True):

    # No triger
    if type_flag == 0: 
        promt_to_use = "\n" + "A:"

    # Use direct triger
    elif type_flag == 1: 
        if args.dataset in ["aqua","commonsensqa",]:
            direct_answer_trigger = "Therefore, among A through E, the answer is"
            promt_to_use = "\n" + direct_answer_trigger
        elif args.dataset in ["gsm8k","addsub","multiarith","svamp","singleeq"]:
            direct_answer_trigger = "Therefore, the answer (arabic numerals) is"
            promt_to_use = "\n" + direct_answer_trigger
        elif args.dataset in ["strategyqa","coin_flip"]:
            direct_answer_trigger = "Therefore, the answer (Yes or No) is"
            promt_to_use = "\n" + direct_answer_trigger
        elif args.dataset in ["bigbench_date"]:
            direct_answer_trigger = "Therefore, among A through F, the answer is"
            promt_to_use = "\n" + direct_answer_trigger
        elif args.dataset in ["object_tracking"]:
            direct_answer_trigger = "Therefore, among A through C, the answer is"
            promt_to_use = "\n" + direct_answer_trigger
        elif args.dataset in ["last_letters"]:
            direct_answer_trigger = "Therefore, the answer is"
            promt_to_use = "\n" + direct_answer_trigger

    # Add cot triger
    elif type_flag == 2: 
        direct_answer_trigger = "A: Let's think step by step."
        promt_to_use = "\n" + direct_answer_trigger
        pass

    # Add constructe template
    elif type_flag == 3: 
        temp_dict = {'instruction':x}
        promt_to_use = PROMPT_DICT["prompt_no_input"].format_map(temp_dict)
        pass

    # Add constructe template, add cot triger
    elif type_flag == 4: 
        x = x + " Let's think first. Stream of consciousness:"
        temp_dict = {'instruction':x}
        promt_to_use = PROMPT_DICT["prompt_no_input"].format_map(temp_dict)
        pass

    # Add constructe template, add cot triger
    elif type_flag == 5: 
        x = x + " Let's think step by step."
        temp_dict = {'instruction':x}
        promt_to_use = PROMPT_DICT["prompt_no_input"].format_map(temp_dict)
        pass

    # Add constructe template, add cot triger
    elif type_flag == 6: 
        x = x + " Let's break it down and consider each step."
        temp_dict = {'instruction':x}
        promt_to_use = PROMPT_DICT["prompt_no_input"].format_map(temp_dict)
        pass

    # Add constructe template, add cot triger
    elif type_flag == 7: 
        x = x + " We should evaluate every stage one at a time."
        temp_dict = {'instruction':x}
        promt_to_use = PROMPT_DICT["prompt_no_input"].format_map(temp_dict)
        pass

    # Add constructe template, add cot triger
    elif type_flag == 8: 
        x = x + " Let's carefully progress through each stage."
        temp_dict = {'instruction':x}
        promt_to_use = PROMPT_DICT["prompt_no_input"].format_map(temp_dict)
        pass

    if with_q:
        x = "Q: " + x + promt_to_use
    else:
        if need_x:
            x = x + promt_to_use
    y = y.strip()

    return x,y,promt_to_use


if __name__ == "__main__":

    dir_name = args.record_save_dir
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    file_name = args.record_special_tag+args.dataset+"-"+str(args.prompt_type)+".txt"
    file_name = os.path.join(dir_name,file_name)

    response_mid = ''
    correct_sum = 0
    start_time = time.time()
    num_instance = int(args.instance_percent*len(questions))
    for i in range(num_instance):

        if args.prompt_type in [0,1]:
            x,y,promt_to_use = get_prompt_simple(questions[i],answers[i],args.prompt_type)
            response_raw = evaluate_raw(x,top_p=args.top_p,max_new_tokens=args.max_new_tokens)
        elif args.prompt_type in [2]:
            x,y,promt_to_use = get_prompt_simple(questions[i],answers[i],args.prompt_type)
            response_mid = evaluate_raw(x,top_p=args.top_p,max_new_tokens=args.max_new_tokens)
            if args.discard_repeat:
                response_mid = discard_repeat_s(response_mid)
            response_mid,y,promt_to_use = get_prompt_simple(response_mid,answers[i],1,with_q=False)
            response_raw = evaluate_raw(response_mid,top_p=args.top_p,max_new_tokens=64)
        elif args.prompt_type in [3,4,5,6,7,8]:
            x,y,promt_to_use = get_prompt_simple(questions[i],answers[i],args.prompt_type,with_q=False,need_x=False)
            x = promt_to_use
            response_mid = evaluate_raw(x,top_p=args.top_p,max_new_tokens=args.max_new_tokens)
            if args.discard_repeat:
                response_mid = discard_repeat_s(response_mid)
            response_mid,y,promt_to_use = get_prompt_simple(response_mid,answers[i],1,with_q=False)
            response_raw = evaluate_raw(response_mid,top_p=args.top_p,max_new_tokens=64)

        response = response_raw.split(promt_to_use)[1].strip()

        if args.dataset in ("aqua", "commonsensqa"):
            pred = re.findall(r'A|B|C|D|E', response)
        elif args.dataset == "bigbench_date":
            pred = re.findall(r'A|B|C|D|E|F', response)
        elif args.dataset in ("object_tracking"):
            pred = re.findall(r'A|B|C', response)
        elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
            pred = response.replace(",", "")
            pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
        elif args.dataset in ("strategyqa", "coin_flip"):
            pred = response.lower()
            pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
            pred = pred.split(" ")
            pred = [i for i in pred if i in ("yes", "no")]
        elif args.dataset == "last_letters":
            pred = re.sub("\"|\'|\n|\.|\s","", response)
            pred = [pred]
        else:
            raise ValueError("dataset is not properly defined ...")
        
        if len(pred) == 0:
            pred = ""
        else:
            pred = pred[0]
        if pred != "":
            if pred[-1] == ".":
                pred = pred[:-1]

        pred_correct = pred == y
        pred_str = "(Extracted Answer:"+pred+"\t Correct:"+str(pred_correct)+")"
        if pred_correct:
            correct_sum += 1

        count_str = "("+str(i+1)+"/"+str(num_instance)+")"
        print("=====================",count_str)
        # print("===(Instruction)===\n", x)
        print("===(Response-Mid)===\n", response_mid.encode('utf-8').decode('utf-8'))
        print("===(Response)===\n", response)
        print("===(Groudtruth)===\n",y)
        print("==="+pred_str+"===\n")
        print()

        with open(file_name,'a') as f:
            f.write("====================="+count_str+"\n")
            # f.write("===(Instruction)===\n"+x+"\n")
            f.write("===(Response-Mid)===\n"+response_mid.encode('utf-8').decode('utf-8')+"\n")
            f.write("===(Response)===\n"+response+"\n")
            f.write("===(Groudtruth)===\n"+y+"\n")
            f.write("==="+pred_str+"===\n")

        pass

    time_used = time.time() - start_time
    print("=====================")
    print("Final Acc:", str(correct_sum/num_instance))
    print("Time Used:", str(time_used/60)+"(min)")

    with open(file_name,'a') as f:
        f.write("=====================\n")
        f.write("Final Acc:" + str(correct_sum/num_instance)+"\n")
        f.write("Time Used:" + str(time_used/60)+"(min)\n")

