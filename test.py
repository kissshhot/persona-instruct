# 假设有一个列表
my_list = [10, 20, 30, 40, 50]

# 假设idx是你要移除的元素的索引
idx = 2  # 例如，移除第三个元素，索引为2

# 使用del语句
del my_list[idx]

# 或者使用pop方法
# my_list.pop(idx)

print(my_list.pop(0))  # 输出: [10, 20, 40, 50]
print(my_list)