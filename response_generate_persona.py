# 这里会让模型对指令生成多个回答，并采用rejection sampling 和 preference-based sampling来筛选出一个最好的回答
import os
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
# os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
model_id = "/data1/dyf/model/Mistral-7B-Instruct-v0.3/"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
from prompts.prompt_template_persona2 import answer_generate_persona
import copy
# rejection sampling
# 接下来，我们设定一些标准，比如：
# 回答必须与问题相关。
# 回答不能含糊或模棱两可。

# preference-based sampling
# 假设我们从过去的用户反馈中了解到用户更喜欢直接、确定性的回答，而不是模棱两可的回答。
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        # required=True,
        default="/home/dyf/data_generate/persona-instruct/data/lima/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        # required=True,
        default="/home/dyf/data_generate/persona-instruct/data/lima/persona_add_lima.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
    )
    parser.add_argument(
        "--batch_length",
        type=int,
        default=200,
        help="ins generated each round",
    )
    parser.add_argument(
        "--roundi",
        type=int,
        default=0,
        help="round",
    )
    parser.add_argument(
        "--th",
        type=float,
        default=5.0,
        help="th of ucb",
    )
    return parser.parse_args()

def output_log_jsonl(log_file, all_logs):
    with open(log_file, "w") as f:
        for log in all_logs:
            f.write(json.dumps(log) + "\n")

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

def persona_response_generate_main(batch_dir, seed_tasks, lima_data, model, sampling_params, chat_formatting_function):
    all_logs = lima_data
    copied_list = copy.deepcopy(seed_tasks)
    for t in copied_list:
        try:
            respondant = t['respondant']
        except:
            continue
        instruction = t['conversations'][0]
        prompt = answer_generate_persona.format(respondant=respondant, instruction=instruction)
        # conversation = [{"role": "user", "content": inputs}]
        # # tools = [get_current_weather]

        # # format and tokenize the tool use prompt 
        # inputs = tokenizer.apply_chat_template(
        #             conversation,
        #             add_generation_prompt=True,
        #             # return_dict=True,
        #             return_tensors="pt",
        # )
        # prompt = t['conversations'][0]
        # inputs = inputs.to('cuda')
        result = use_vllm([prompt], model, sampling_params, chat_formatting_function).strip()
        # answer = result.split('### Response:')[1]
        t['conversations'].append(result)
        all_logs.append(t)
        # if len(all_logs) >= 10000:
        #     break
        output_log_jsonl(os.path.join(batch_dir, "persona_response.jsonl"), all_logs) 

def multi_sample(seed_tasks): #生成多个回答并进行选择
    for t in seed_tasks:
        respondent = t['respondent']
        instruction = t['question']
        inputs = answer_generate.format(respondent=respondent, instruction=instruction)
        conversation = [{"role": "user", "content": inputs}]
        # tools = [get_current_weather]

        # format and tokenize the tool use prompt 
        inputs = tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    # return_dict=True,
                    return_tensors="pt",
        )

        inputs = inputs.to('cuda')
        outputs = model.generate(inputs, max_new_tokens=5000, do_sample=True, temperature=0.7, top_p=0.9) #现在貌似是gs，后面可能要改成sample
        result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        answer = result.split('### Response:')[1]
        t['conversations'].append(answer)
        output_log_jsonl(os.path.join("/home/dyf/data_generate/persona-instruct/data/lima/persona_response/", f"new_data.jsonl"), seed_tasks) 

# if __name__ == "__main__":
#     args = parse_args()
#     seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
#     single_sample(seed_tasks)
