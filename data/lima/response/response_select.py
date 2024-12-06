import json

# 文件路径
file_path = '/home/dyf/data_generate/persona-instruct/data/lima/response/response_score.jsonl'

# 要找的键
key = 'rewards_score'

# 读取文件并存储数据
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line))

# 按照键值排序
data_sorted = sorted(data, key=lambda x: x.get(key, 0), reverse=True)

# 选出键值最大的1万条
top_10k = data_sorted[:10000]

# # 打印结果或保存到文件
# for item in top_10k:
#     print(item)

# 如果需要保存到文件
with open('/home/dyf/data_generate/persona-instruct/data/lima/response/response-top1w.jsonl', 'w', encoding='utf-8') as file:
    for item in top_10k:
        file.write(json.dumps(item, ensure_ascii=False) + '\n')