import numpy as np
import pandas as pd
from tqdm import tqdm
import collections
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
from utils.distance import Euclid_d, Maharanovis_d, Manhattan_d
import argparse


# 引数の設定
parser = argparse.ArgumentParser(description="Process knn")
parser.add_argument(
    "-d", "--distance", default="Eu", help="select distance ['Eu','Man','Mah']"
)
parser.add_argument("-o", "--output", default="output.png", help="output file name")


class KNN:
    def __init__(self):
        # dataの読み込み
        self.df = pd.read_csv("../data/iris.csv")
        self.df_data = self.df.loc[:, "sepal_length":"petal_width"]
        self.data = self.df_data.iloc[:].values
        # 共分散行列
        self.A = np.cov(self.data.T)

    def predict(self, k, dist_type):
        acc_list = []
        # leave-one-out method
        for i, d in enumerate(self.data):
            test_data = d
            test_label = self.df.iloc[i, 4]
            train_data = np.delete(self.data, i, 0)
            train_label = self.df.iloc[self.df.index != i, 4].values
            test_data = np.array([test_data for j in range(len(train_data))])
            if dist_type == "Eu":
                dist = Euclid_d(test_data, train_data)
            elif dist_type == "Man":
                dist = Manhattan_d(test_data, train_data)
            elif dist_type == "Mah":
                dist = Maharanovis_d(test_data, train_data, self.A)
            dic = {key: val for key, val in zip(dist, train_label)}
            dic = sorted(dic.items())
            # 距離順にk個のラベルをリストにする
            preds = list(map(lambda x: x[1], dic[:k]))
            # ｋこのラベルの中で一番多いものがpred
            c = collections.Counter(preds)
            pred = c.most_common()[0][0]
            # predがtestlabelだったら1
            if pred == test_label:
                acc_list.append(1)
            # 間違ったら0
            else:
                acc_list.append(0)
        return sum(acc_list) / len(acc_list)


if __name__ == "__main__":
    args = parser.parse_args()
    distance = args.distance
    file_name = args.output
    knn = KNN()
    K_acc_list = []
    for k in tqdm(range(1, 31)):
        acc = knn.predict(k, distance)
        K_acc_list.append(acc)
    x = range(1, 31)
    y = K_acc_list
    fig = plt.figure()
    plt.plot(x, y)
    plt.title(distance + " distance")
    plt.xlabel("k")
    plt.ylabel("accuracy")
    fig.savefig(file_name)
