import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import umap.umap_ as umap

# 生成示例数据
# 假设我们有1000个样本，每个样本有50个特征
X, _ = make_blobs(n_samples=1000, n_features=50, centers=30, random_state=42)

# 使用UMAP将数据降到2D
reducer = umap.UMAP(n_components=2, n_neighbors=100, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X)

# 使用K-Means进行聚类
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X_umap)

# 绘制聚类散点图
plt.figure(figsize=(12, 8))
for i in range(n_clusters):
    cluster_points = X_umap[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}', s=10)

plt.title("Cluster Scatter Plot using UMAP and KMeans")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.legend()
plt.savefig("/home/dyf/data_generate/persona-instruct/figure/cluster_scatter_plot.png", dpi=300)  # dpi=300确保图像质量
plt.show()