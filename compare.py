import json
path_1 = ''
path_2 = ''
seed_tasks = [json.loads(l) for l in open(path_1, "r")]
com_tasks = [json.loads(l) for l in open(path_1, "r")]

for idx in range(len(seed_tasks)):
    if seed_tasks[idx]['complexity_score'] >= com_tasks[idx]['complexity_score']:
        final_logs.append(seed_tasks[idx])
    else:
        pre_logs.append(com0[idx])
    main_com(roundi, pre_logs, args.is_vllm,args.batch_length, model, sampling_params, chat_formatting_function, documents)