import numpy as np
from datasketch import MinHash, MinHashLSH

# 创建 MinHashLSH 对象
lsh = MinHashLSH(threshold=0.9, num_perm=128)

# 假设已有文档
documents = [
    "这是第一篇文档。",
    "这是第二篇文档。",
    "这是第三篇文档。",
]

# 存储每个文档的 MinHash
minhashes = {}

for i, doc in enumerate(documents):
    m = MinHash()
    for word in doc.split():
        m.update(word.encode('utf8'))
    lsh.insert(f'doc_{i}', m)
    minhashes[f'doc_{i}'] = m

# 新加入的文档
new_doc = "这是第一篇文档！"

# 为新文档计算 MinHash
new_minhash = MinHash()
for word in new_doc.split():
    new_minhash.update(word.encode('utf8'))

# 查询相似的文档
result = lsh.query(new_minhash)

# 判断相似度
if len(result) == 0:
    print("新文档满足相似度阈值，可以加入。")
    documents.append(new_doc)
else:
    print("新文档与已有文档相似，不加入。")

print("当前文档列表：")
for doc in documents:
    print(doc)
