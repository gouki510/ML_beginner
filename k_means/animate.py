import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from distance import Euclid_d

# dataの読み込み
df = pd.read_csv("iris.csv")
df_data = df.loc[:, "sepal_length":"petal_width"]
data = df_data.iloc[:].values

fig = plt.figure()
target_plot = fig.add_subplot(121)
k_means_plot = fig.add_subplot(122)
k_means_plot.grid()
# 主成分分析
pca = PCA(2)
pca.fit(data)
# データを主成分空間に写像
feature = pca.transform(data)
color_list1 = []
for i in list(df.iloc[:, 4]):
    if i == "setosa":
        color_list1.append(0)
    elif i == "versicolor":
        color_list1.append(1)
    elif i == "virginica":
        color_list1.append(2)
target_plot.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=color_list1)
target_plot.grid()
target_plot.set_title("PCA")


first = np.random.choice(len(data), 3)
centroids = data[first]
epoch = -1


def update(j):
    k = 3
    data_size, n_features = data.shape
    new_centroids = np.zeros((k, n_features))
    cluster = np.zeros(data_size)
    global centroids
    global first
    global epoch
    for i in range(data_size):
        dist = Euclid_d(centroids, data[i])
        cluster[i] = np.argsort(dist)[0]
    for j in range(k):
        new_centroids[j] = data[cluster == j].mean(axis=0)
    centroids = new_centroids
    for f in range(k):
        k_means_plot.plot(feature[first[k], 0], feature[first[k], 1], "*")
    k_means_plot.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=cluster)
    epoch += 1
    k_means_plot.set_title("epoch:" + str(epoch))


ani = FuncAnimation(fig, update, frames=30, interval=500)

ani.save("k-means.mp4", writer="ffmpeg")
