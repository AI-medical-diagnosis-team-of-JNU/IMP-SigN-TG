from sklearn.cluster import KMeans
import sklearn.metrics as metrics

class K_means():
    def __init__(self, data):
        self.x = data
    def best_k(self):
        score_list = list()  # 用来存储每个K下模型的平局轮廓系数
        silhouette_int = -1  # 初始化的平均轮廓系数阀值
        for n_clusters in range(2, 10):  # 遍历从2到10几个有限组
            model_kmeans = KMeans(n_clusters=n_clusters, random_state=0)  # 建立聚类模型对象
            cluster_labels_tmp = model_kmeans.fit_predict(self.x)  # 训练聚类模型
            silhouette_tmp = metrics.silhouette_score(self.x, cluster_labels_tmp)  # 得到每个K下的平均轮廓系数
            if silhouette_tmp > silhouette_int:  # 如果平均轮廓系数更高
                best_k = n_clusters  # 将最好的K存储下来
                silhouette_int = silhouette_tmp  # 将最好的平均轮廓得分存储下来
                best_kmeans = model_kmeans  # 将最好的模型存储下来

                cluster_labels_k = cluster_labels_tmp  # 将最好的聚类标签存储下来
            score_list.append([n_clusters, silhouette_tmp])  # 将每次K及其得分追加到列表
        return best_k, best_kmeans


import numpy as np
x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
X=np.array(list(zip(x1,x2))).reshape(len(x1),2)
print(X)
if __name__ == '__main__':
    print(X.shape)
    model = K_means(X)
    print(model.best_k())