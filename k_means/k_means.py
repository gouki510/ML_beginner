# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from utils.distance import Euclid_d, Maharanovis_d, Manhattan_d
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # 主成分分析器


class Kmeans:
    def __init__(self, df):
        # dataの読み込み
        self.df = df
        self.df_data = self.df.loc[:, "sepal_length":"petal_width"]
        self.data = self.df_data.iloc[:].values
        # 共分散行列
        self.A = np.cov(self.data.T)
        # 主成分分析
        self.pca = PCA(2)
        self.pca.fit(self.data)
        # データを主成分空間に写像
        self.feature = self.pca.transform(self.data)
        pd.DataFrame(self.feature).to_csv("PCA.csv")
        self.color_list1 = []
        for i in list(self.df.iloc[:, 4]):
            if i == "setosa":
                self.color_list1.append(0)
            elif i == "versicolor":
                self.color_list1.append(1)
            elif i == "virginica":
                self.color_list1.append(2)
        self.fig1 = plt.figure(figsize=(6, 6))
        plt.scatter(
            self.feature[:, 0], self.feature[:, 1], alpha=0.8, c=self.color_list1
        )
        plt.grid()
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA")

    def main(self, k, dist_type):
        data_size, n_features = self.data.shape
        centroids = self.data[np.random.choice(len(self.data), k)]
        new_centroids = np.zeros((k, n_features))
        cluster = np.zeros(data_size)
        for epoch in range(300):
            for i in range(data_size):
                if dist_type == "Eu":
                    dist = Euclid_d(centroids, self.data[i])
                elif dist_type == "Man":
                    dist = Manhattan_d(centroids, self.data[i])
                elif dist_type == "Mah":
                    dist = Maharanovis_d(centroids, self.data, A)
                cluster[i] = np.argsort(dist)[0]
            for j in range(k):
                new_centroids[j] = self.data[cluster == j].mean(axis=0)
            if np.sum(new_centroids == centroids) == k:
                print("break")
                break
            centroids = new_centroids
        return cluster

    def plot(self):
        fig2 = plt.figure(figsize=(6, 6))
        plt.scatter(self.feature[:, 0], self.feature[:, 1], alpha=0.8, c=cluster)
        plt.grid()
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Euclid distance")
        self.fig1.savefig("PCA.png")
        fig2.savefig("test.png")


if __name__ == "__main__":
    df = pd.read_csv("iris.csv")
    kmeans = Kmeans(df)

    cluster = kmeans.main(3, "Eu")
    kmeans.plot()
