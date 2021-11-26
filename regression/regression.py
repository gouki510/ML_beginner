import numpy as np
from numpy import linalg as LA
import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Process Regression")
parser.add_argument("-f", "--features", action="append", help="select feture")


class Regression:
    def __init__(self, features):
        df = pd.read_csv("../data/auto-mpg.csv")
        self.features = list(map(str, features))
        print("selected fetures:", self.features)
        self.X_data = df.loc[:, self.features].values
        self.Y_data = df.loc[:, "mpg"].values
        self.before_loss = 0
        self.after_loss = 0
        if "horsepower" in self.features:
            idx = self.features.index("horsepower")
            self.X_data, self.Y_data = self.preprocess(self.X_data, self.Y_data, idx)

    def preprocess(self, X_data, Y_data, idx):
        new_data = []

        new_Y = []
        for data, t in zip(X_data, Y_data):
            if data[idx] == "?":
                continue
            else:
                temp_data = []
                for i in range(len(self.features)):
                    if i == idx:
                        temp_data.append(int(data[i]))
                    else:
                        temp_data.append(data[i])
                new_data.append(temp_data)
                new_Y.append(t)
        return np.array(new_data), np.array(new_Y)

    def predict(self):

        num_data, n_feature = self.X_data.shape
        X_data = self.X_data.astype("float")
        b = np.ones((num_data, 1))
        X_data = np.hstack((X_data, b))

        # 重みとバイアスの初期化
        W = np.random.randn(n_feature + 1)
        pred1 = np.dot(X_data, W.T)
        loss1 = LA.norm(self.Y_data - pred1) / 2
        self.before_loss = loss1

        # Wを更新 ムーア・ペローンズの疑似逆行列
        A_T = LA.inv(np.dot(X_data.T, X_data))
        W = np.dot(np.dot(A_T, X_data.T), self.Y_data)
        pred2 = np.dot(X_data, W.T)
        loss2 = loss = LA.norm(self.Y_data - pred2) / 2
        self.after_loss = loss2


if __name__ == "__main__":
    args = parser.parse_args()
    features = args.features
    model = Regression(features)
    model.predict()
    print("L2 loss", model.after_loss)
