import numpy as np
import torch
import json
from transformers import AutoModel, AutoTokenizer
from scipy.stats import entropy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# N = 1000

tokenizer_embedding = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
model_embedding = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5') # , device_map={"": "cuda"}
model_embedding.eval()
seed_tasks_path = "/home/dyf/data_generate/persona-instruct/data/lima/merged/diff_merged_instruct_20000_person2_round_0.jsonl"
seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
persona_doc = []
# for tmp in seed_tasks:
#     persona_doc.append(tmp['questioner'])
for tmp in seed_tasks:
    persona_doc.append(tmp['conversations'][0])

result = []
for i in range(0, len(seed_tasks), 1000):
    # Tokenize sentences
    encoded_input = tokenizer_embedding(persona_doc[i:i+1000], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model_embedding(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        txt_embeddings = model_output[0][:, 0]
    # normalize embeddings
    txt_embeddings = torch.nn.functional.normalize(txt_embeddings, p=2, dim=1)
    if i == 0:
        old_embeddings = txt_embeddings
    if i > 0:
        old_embeddings = torch.cat((old_embeddings, txt_embeddings), dim=0)
    # 计算点积矩阵
    dot_product_matrix = old_embeddings @ old_embeddings.T  # 形状为 (N, N)

    # 求和所有点积的结果
    # 只需要上三角部分的和，不包括对角线（自点积），使用 torch.triu_indices
    triu_indices = torch.triu_indices(dot_product_matrix.size(0), dot_product_matrix.size(1), 1)
    total_sum = dot_product_matrix[triu_indices].sum()

    # 计算平均值
    num_pairs = triu_indices.size(1)  # 上三角部分的元素数量
    average = total_sum / num_pairs
    result.append(average)
    print("所有向量两两点积的平均值:", average.item())

print(result)
x = range(len(result))  # X轴数据
y = result  # Y轴数据

# 创建一个图形
plt.figure()

# 绘制折线图
plt.plot(x, y, marker='o')  # 'o'表示用圆圈标记每个数据点

# 在每个数据点上标注数值
for i in range(len(x)):
    plt.text(x[i], y[i], f'{y[i]}', ha='center', va='bottom')  # ha和va用于调整文本位置

# 添加标题和标签
plt.title('折线统计图')
plt.xlabel('X轴')
plt.ylabel('Y轴')


# 显示图形
plt.show()
plt.savefig(f"/home/dyf/data_generate/persona-instruct/ablation_test/distance_result.png", format='png', dpi=300)




# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity


# cosine_similarities = cosine_similarity(txt_embeddings)
# modified_similarities = 1 - cosine_similarities
# # 计算相似度矩阵中所有元素的和
# total_diversity_sum = np.sum(modified_similarities)
# avg_score = total_diversity_sum / (N * (N - 1))
# print("diversity score:", avg_score)