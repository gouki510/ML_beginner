import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance
import collections
import matplotlib.pyplot as plt
from distance import Euclid_d, Maharanovis_d, Manhattan_d

# dataの読み込み
df = pd.read_csv("iris.csv")
df_data = df.loc[:, "sepal_length":"petal_width"]
data = df_data.iloc[:].values

# 共分散行列
A = np.cov(data.T)


def main(k, data, dist_type):
    acc_list = []
    # leave-one-out method
    for i, d in enumerate(data):
        test_data = d
        test_label = df.iloc[i, 4]
        train_data = np.delete(data, i, 0)
        train_label = df.iloc[df.index != i, 4].values
        test_data = np.array([test_data for j in range(len(train_data))])
        if dist_type == "Eu":
            dist = Euclid_d(test_data, train_data)
        elif dist_type == "Man":
            dist = Manhattan_d(test_data, train_data)
        elif dist_type == "Mah":
            dist = Maharanovis_d(test_data, train_data, A)
        dic = {key: val for key, val in zip(dist, train_label)}
        dic = sorted(dic.items())
        preds = list(map(lambda x: x[1], dic[:k]))
        c = collections.Counter(preds)
        pred = c.most_common()[0][0]
        if pred == test_label:
            acc_list.append(1)
        else:
            acc_list.append(0)
    return sum(acc_list) / len(acc_list)


if __name__ == "__main__":
    K_acc_list = []
    for k in tqdm(range(1, 31)):
        K_acc_list.append(main(k, data, "Mah"))
    x = range(1, 31)
    y = K_acc_list
    fig = plt.figure()
    plt.plot(x, y)
    plt.title("Maharanovis distance")
    plt.xlabel("k")
    plt.ylabel("accuracy")
    fig.savefig("result3.png")
