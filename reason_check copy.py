import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import pdb
import torch
from datasketch import MinHash, MinHashLSH
import numpy as np
import os
import json
import vllm
from importlib import import_module
import random
import re
import string
from tqdm import tqdm
import argparse
from prompts.prompt_template_persona2 import persona_generate, persona_generate_simple, persona_diff_instruct_generate_wo_persona, persona_diff_instruct_generate_simple, persona_diff_instruct_generate_re, persona_com_instruct_generate_rewrite_wo_persona
from prompts.score_template import score_template
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
model_id = "/data1/dyf/model/Mistral-7B-Instruct-v0.3/"
tokenizer = AutoTokenizer.from_pretrained(model_id)
def dynamic_import_function(function_path):
    '''
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    templates.create_prompt_with_huggingface_tokenizer_template
    '''
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function

def create_prompt_with_huggingface_tokenizer_template(messages, tokenizer, add_bos=False):
    formatted_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if add_bos:
        formatted_text = tokenizer.bos_token + formatted_text
    return formatted_text

chat_formatting_function = dynamic_import_function("templates.create_prompt_with_huggingface_tokenizer_template")
model = vllm.LLM(
    model=model_id,
    tokenizer=model_id,
    tokenizer_mode="auto",
    tensor_parallel_size=torch.cuda.device_count(),
    tokenizer_revision=None, 
    revision=None,
)

sampling_params = vllm.SamplingParams(
    temperature=0.0,  # greedy decoding
    max_tokens=5000,
    # stop=args.additional_stop_sequence,
    # --additional_stop_sequence',
    # type=str,
    # nargs="+",
    # default=[],
)

def use_vllm(prompts, model, sampling_params, chat_formatting_function):
    
    # chat_formatting_function = dynamic_import_function("templates.create_prompt_with_huggingface_tokenizer_template")
    # model = vllm.LLM(
    #     model=model_id,
    #     tokenizer=model_id,
    #     tokenizer_mode="auto",
    #     tensor_parallel_size=torch.cuda.device_count(),
    #     tokenizer_revision=None, 
    #     revision=None,
    # )
    
    # sampling_params = vllm.SamplingParams(
    #     temperature=0.7,  # greedy decoding
    #     max_tokens=5000,
    #     # stop=args.additional_stop_sequence,
    #     # --additional_stop_sequence',
    #     # type=str,
    #     # nargs="+",
    #     # default=[],
    # )
    # apply chat formatting
    formatted_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
        formatted_prompts.append(formatted_prompt)
    prompts = formatted_prompts
            
    outputs = model.generate(prompts, sampling_params)
    outputs = [it.outputs[0].text for it in outputs]
    return outputs[0]

# 指定你的JSON文件路径
file_path = '/home/dyf/data_generate/persona-instruct/data/lima/share_gpt/persona-instruct.json'

# 使用with语句打开文件，确保文件会被正确关闭
with open(file_path, 'r', encoding='utf-8') as file:
    # 将文件内容解析为Python字典
    datas = json.load(file)

# 检查conversations列表是否非空，并获取第一个元素
for data in datas:
    if data.get('conversations') and len(data['conversations']) > 0:
        first_conversation = data['conversations'][0]
        # 检查content字段是否包含\n\nReason:
        if '\n\nReason:' in first_conversation.get('content', ''):
            prompt = first_conversation['content'].split('\n\nReason:')[0]
            # prompt = first_conversation # answer_generate.format(instruction=instruction).strip()
            # conversation = [{"role": "user", "content": inputs}]
            # tools = [get_current_weather]

            # format and tokenize the tool use prompt 
            # inputs = tokenizer.apply_chat_template(
            #             conversation,
            #             add_generation_prompt=True,
            #             # return_dict=True,
            #             return_tensors="pt",
            # )

            # inputs = inputs.to('cuda')
            result = use_vllm([prompt], model, sampling_params, chat_formatting_function).strip()
            # outputs = model.generate(inputs, max_new_tokens=5000, do_sample=True, temperature=0.7, top_p=0.9) #现在貌似是gs，后面可能要改成sample
            # result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            # try:
            #     answer = result.split('### Response:')[1]
            # except:
            #     continue
            # answer = result
            data['conversations'][0]['content'] = prompt
            data['conversations'][1]['content'] = result
            output_file = '/home/dyf/data_generate/persona-instruct/data/lima/share_gpt/persona.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(datas, f, ensure_ascii=False, indent=2)