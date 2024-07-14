from sklearn.datasets import load_wine
import numpy as np 
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import os

def get_kernel_function(kernel:str, index = 2):
    # 线性核，计算两个向量的点积dot
    if kernel == "linear":
        return lambda x, y: np.dot(x, y)
    # 多项式核，先取点积再加指数，这里默认2次
    elif kernel == "polynomial":
        return lambda x, y: (np.dot(x, y) + 1)**index  
    # 径向基函数核，非线性
    elif kernel == "rbf":
        def rbf_kernel(x, y, gamma=0.05):
            return np.exp(-gamma * np.linalg.norm(x - y) ** 2)
        return rbf_kernel
    else:
        return None

class PCA:
    def __init__(self, n_components:int=2, kernel:str="rbf") -> None:
        self.n_components = n_components
        self.kernel_f = get_kernel_function(kernel)

    def fit(self, X:np.ndarray):
        # 根据文档给出的步骤：
        # 计算平均值，axis表示沿着行方向进行操作，即计算每列的平均值
        self.mean = np.mean(X, axis=0)
        # 计算中心矩阵
        X_center = X - self.mean
        # 计算协方差矩阵
        cov_matrix = np.cov(X_center, rowvar=False)
        # 计算特征值和特征向量
        special_values, special_vectors = np.linalg.eig(cov_matrix)
        # 对特征值进行升序排序
        sort_index = np.argsort(special_values)
        # 取特征值前二的特征向量，即倒数后两个，因为升序
        self.sepcial_matrix = special_vectors[:, sort_index[:-self.n_components-1:-1]]
        return self
    
    # 将原数据投影
    def transform(self, X:np.ndarray):
        # 进行矩阵乘法降维
        X_center = X - self.mean
        return np.dot(X_center, self.sepcial_matrix)

class KMeans:
    def __init__(self, n_clusters:int=3, max_iter:int=10) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = None
        self.labels = None

    # 初始化中心
    def initialize_centers(self, points):
        sample_count, div = points.shape
        self.centers = np.zeros((self.n_clusters, div))
        for i in range(self.n_clusters):
            # 从样本的下标中随机抽样，抽取10个，不可重复
            random_index = np.random.choice(sample_count, size=10, replace=False)
            # 将抽取的10个样本取均值，作为中心
            self.centers[i] = points[random_index].mean(axis=0)
        return self.centers
    
    # 对每个样本进行划分
    def assign_points(self, points):
        point_count = points.shape[0]
        center_count = self.centers.shape[0]
        # 初始化标签
        self.labels = np.zeros(point_count, dtype=int)  
        # 遍历每个点的下标
        for i in range(point_count):  
            # 初始化最近距离和最近中心
            min_distance = float('inf')  
            closest_center = -1  
            # 遍历每个中心
            for j in range(center_count):  
                # 计算欧式距离
                distance = np.sqrt(((points[i] - self.centers[j])**2).sum()) 
                if distance < min_distance:
                    # 更新最近点
                    min_distance = distance
                    closest_center = j
            # 将最近的中心索引赋值给当前点的标签
            self.labels[i] = closest_center  
        return self.labels

    # 更新中心点
    def update_centers(self, points):
        # 将每个簇的点取平均得到新中心点
        for i in range(self.n_clusters):
            self.centers[i] = np.mean(points[self.labels == i], axis=0)

    # 迭代更新
    def fit(self, points):
        self.initialize_centers(points)
        for _ in range(self.max_iter):
            self.assign_points(points)
            # 保留之前的中心
            before_centers = self.centers.copy()
            # 更新中心
            self.update_centers(points)
            # 如果之前的中心和现在的中心完全一样，停止KMeans
            if np.all(before_centers == self.centers):
                break
        return self

    # 对降维之后的数据进行KMeans聚类
    def predict(self, points):
        return self.assign_points(points)
    
os.chdir(r'C:\Users\86153\Desktop\AI\Lab\lab2\ai_2024sp_exp2_final\part_1')
def load_data():
    words = [
        'computer', 'laptop', 'minicomputers', 'PC', 'software', 'Macbook',
        'king', 'queen', 'monarch','prince', 'ruler','princes', 'kingdom', 'royal',
        'man', 'woman', 'boy', 'teenager', 'girl', 'robber','guy','person','gentleman',
        'banana', 'pineapple','mango','papaya','coconut','potato','melon',
        'shanghai','HongKong','chinese','Xiamen','beijing','Guilin',
        'disease', 'infection', 'cancer', 'illness', 
        'twitter', 'facebook', 'chat', 'hashtag', 'link', 'internet',
    ]
    w2v = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary = True)
    vectors = []
    for w in words:
        vectors.append(w2v[w].reshape(1, 300))
    vectors = np.concatenate(vectors, axis=0)
    return words, vectors

if __name__=='__main__':
    words, data = load_data()
    pca = PCA(n_components=2).fit(data)
    data_pca = pca.transform(data)

    kmeans = KMeans(n_clusters=7).fit(data_pca)
    clusters = kmeans.predict(data_pca)

    # 得到聚类图像
    plt.figure()
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters)
    for i in range(len(words)):
        plt.annotate(words[i], data_pca[i, :]) 
    plt.title("PB21111686_zz")
    plt.savefig("PCA_KMeans.png")