# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from utils.distance import Euclid_d, Maharanovis_d, Manhattan_d
import random
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="K means Process")
parser.add_argument("-k", "--k", default=3, help="k")
parser.add_argument("-d", "--distance", default="Eu", help="select distace [Eu,Man,Mah]")
parser.add_argument("-o", "--output", default="output.png")


class Kmeans:
    def __init__(self):
        # dataの読み込み
        self.df = pd.read_csv("../data/iris.csv")
        self.df_data = self.df.loc[:, "sepal_length":"petal_width"]
        self.data = self.df_data.iloc[:].values
        # 共分散行列
        self.A = np.cov(self.data.T)
        # 色のリスト作成
        self.color_list1 = []
        for i in list(self.df.iloc[:, 4]):
            if i == "setosa":
                self.color_list1.append(0)
            elif i == "versicolor":
                self.color_list1.append(1)
            elif i == "virginica":
                self.color_list1.append(2)

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

    def plot(self, file_name, dist_type,cluster):
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(
            self.df["petal_length"],
            self.df["petal_width"],
            c=cluster,
            s=50,
            alpha=0.5,
        )
        plt.grid()
        plt.xlabel("petal_length")
        plt.ylabel("petal_width")
        plt.title(dist_type+" distance")
        fig.savefig(file_name)


if __name__ == "__main__":
    args = parser.parse_args()
    k = args.k
    distance = args.distance
    file_name = args.output
    kmeans = Kmeans()
    cluster = kmeans.main(k, distance)
    kmeans.plot(file_name,distance,cluster)
