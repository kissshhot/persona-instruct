# pip install datasketch
from datasketch import MinHash, MinHashLSH

# 假设我们有多个文档或者文本数据
documents = [
    "The quick brown fox jumps over the lazy dog",
    "The quick brown fox jumps over the lazy dog!",
    "A fast brown fox leaps over a sleepy dog",
    "Lorem ipsum dolor sit amet"
]

# 创建一个 MinHashLSH 结构
lsh = MinHashLSH(threshold=0.5, num_perm=128)

# 用于存储每个文档的 MinHash 对象
minhashes = []

for doc_id, doc in enumerate(documents):
    # 创建 MinHash 对象
    m = MinHash(num_perm=128)
    # 将每个单词加入到 MinHash 中（也可以按字符片段处理）
    for word in doc.split():
        m.update(word.encode('utf8'))
    # 将 MinHash 对象插入到 LSH 结构中
    lsh.insert(f"doc{doc_id}", m)
    minhashes.append((doc_id, m))

# 检测相似的文档
for i, m in minhashes:
    result = lsh.query(m)  # 查找与该文档相似的文档
    print(f"Document {i} is similar to {result}")