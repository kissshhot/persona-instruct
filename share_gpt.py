import json

def convert_jsonl_to_sharegpt(input_file, output_file):
    # 存储所有的对话
    sharegpt_data = []

    # 读取 JSONL 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # 解析 JSONL 中的每一行
            data = json.loads(line)
            
            # 创建新的对话结构
            conversation_data = {
                "id": f"conversation_{i + 1}",  # 给每个对话一个唯一的ID
                "conversations": []
            }

            # 处理 conversations 字段
            conversations = data.get("conversations", [])
            if conversations:
                # 偶数索引是提问者（"user"），奇数索引是回答者（"assistant"）
                for j, text in enumerate(conversations):
                    if j > 1:
                        print(i)
                    role = "user" if j % 2 == 0 else "assistant"
                    conversation_data["conversations"].append({
                        "role": role,
                        "content": text.strip()
                    })

            # 添加到总对话列表中
            sharegpt_data.append(conversation_data)

    # 将转换后的数据写入新的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

# 使用该函数
input_file = "/home/dyf/data_generate/persona-instruct/data/lima/response/response-top1w.jsonl"  # 输入的 JSONL 文件
output_file = "/home/dyf/data_generate/persona-instruct/data/lima/share_gpt/persona-instruct.json"  # 输出的 ShareGPT 格式 JSON 文件
convert_jsonl_to_sharegpt(input_file, output_file)