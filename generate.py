import argparse
import json
import os
from persona_com_instruct_generate_demo_lima_persona2 import main_com
from persona_diff_instruct_generate_demo_lima_persona2_filter import main_diff
from persona_respondant_generate_lima_persona2 import respondant_generate_main
from response_generate import response_generate_main
from response_generate_persona import persona_response_generate_main
from filter import embedding_filter_main
import vllm
from importlib import import_module
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# from persona_diff_instruct_generate_demo_lima_persona2 import main_diff
model_id = "/data1/dyf/model/Mistral-7B-Instruct-v0.3/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        # required=True,
        default="/home/dyf/data_generate/persona-instruct/data/lima/epoch/com/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        # required=True,
        default="/home/dyf/data_generate/persona-instruct/data/lima/merged/diff_merged_instruct_12000_person2_round_0.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
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
    parser.add_argument(
        "--is_vllm",
        action="store_true",
        # required=True,
        # default=True,
        # help="The path to the human written data.",
    )
    parser.add_argument(
        "--batch_length",
        type=int,
        default=10,
        help="ins generated each round",
    )
    return parser.parse_args()

def output_log_jsonl(log_file, all_logs):
    with open(log_file, "w") as f:
        for log in all_logs:
            f.write(json.dumps(log) + "\n")

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

if __name__ == "__main__":
    args = parse_args()
    args.is_vllm = True
    all_logs = []
    roundi = args.roundi
    batch_dir = args.batch_dir
    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    lima_data = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    # seed_tasks = [json.loads(l) for l in open('/home/dyf/data_generate/persona-instruct/data/lima/epoch/diff/diff_new_instruct_1030_person2_round_0.jsonl', "r")]
    documents = []
    for tmp in seed_tasks:
        documents.append(tmp['conversations'][0])
    if args.is_vllm == True:
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
    # log2, documents = main_diff(roundi, seed_tasks, args.is_vllm, args.batch_length, model, sampling_params, chat_formatting_function)
    # log1 = main_com(roundi, seed_tasks, args.is_vllm, model, sampling_params, chat_formatting_function, documents)


    seed_tasks, documents = main_diff(roundi, seed_tasks, args.is_vllm, args.batch_length, model, sampling_params, chat_formatting_function)
    # seed_tasks = embedding_filter_main(seed_tasks, args.batch_length)
    # for roundi in range(2):
    # # # roundi = 1
    #     seed_tasks = main_com(roundi, seed_tasks, args.is_vllm,args.batch_length, model, sampling_params, chat_formatting_function, documents)

    # output_log_jsonl(os.path.join("/home/dyf/data_generate/persona-instruct/data/lima/final/", f"final.jsonl"), seed_tasks)

    # response_generate_main(batch_dir, seed_tasks, lima_data, model, sampling_params, chat_formatting_function)
    # seed_tasks = respondant_generate_main(seed_tasks, model, sampling_params, chat_formatting_function)
    # persona_response_generate_main(batch_dir, seed_tasks, lima_data, model, sampling_params, chat_formatting_function)

