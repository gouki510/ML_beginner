# -*- coding:utf-8 -*-
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

sys.path.append("../")
import numpy as np
from utils.distance import Euclid_d
from k_means import Kmeans


class TkGUI:
    def __init__(self):
        self.root = tk.Tk()

        self.canvas1 = tk.Canvas(self.root, width=400, height=300, relief="raised")
        self.canvas1.grid(row=0, column=0)
        self.label1 = tk.Label(self.root, text="k-Means Clustering")
        self.label1.config(font=("helvetica", 14))
        self.canvas1.create_window(200, 25, window=self.label1)
        self.label2 = tk.Label(self.root, text="Type Number of Clusters:")
        self.label2.config(font=("helvetica", 8))
        self.canvas1.create_window(200, 120, window=self.label2)
        self.entry1 = tk.Entry(self.root)
        self.canvas1.create_window(200, 140, window=self.entry1)
        self.df = pd.read_csv("../data/iris.csv")
        self.centroids = np.array([0])
        self.figure1 = plt.Figure(figsize=(4, 3), dpi=100)
        self.scatter1 = FigureCanvasTkAgg(self.figure1, self.root)

    def getcsv(self):
        import_file_path = filedialog.askopenfilename()
        read_file = pd.read_csv(import_file_path)
        df = DataFrame(read_file)

    def makegui(self):
        self.processButton = tk.Button(
            text=" Process k-Means ",
            command=self.getKMeans,
            bg="brown",
            fg="white",
            font=("helvetica", 10, "bold"),
        )
        self.canvas1.create_window(200, 170, window=self.processButton)
        self.root.mainloop()

    def getKMeans(self):
        self.km = Kmeans()
        self.k = int(self.entry1.get())
        self.data = self.km.df.loc[:, "sepal_length":"petal_width"].values
        self.data_size, self.n_features = self.km.data.shape
        if self.centroids.shape[0] == 1:
            self.centroids = self.data[np.random.choice(len(self.data), self.k)]
        self.new_centroids = np.zeros((self.k, self.n_features))
        self.cluster = np.zeros(self.data_size)
        for i in range(self.data_size):
            dist = Euclid_d(self.centroids, self.data[i])
            self.cluster[i] = int(np.argsort(dist)[0])
        for j in range(self.k):
            self.new_centroids[j] = self.data[self.cluster == j].mean(axis=0)
        self.centroids = self.new_centroids
        ax1 = self.figure1.add_subplot(211)
        ax2 = self.figure1.add_subplot(212)
        ax1.tick_params(axis="x", labelsize=7)
        xlabel = "petal_length"
        ylabel = "petal_width"
        xticklabels = ax1.get_xticklabels()
        yticklabels = ax1.get_yticklabels()
        ax1.set_xticklabels(xticklabels, fontsize=7)
        ax1.set_yticklabels(yticklabels, fontsize=7)
        ax1.set_xlabel(xlabel, fontsize=7, loc="right")
        ax1.set_ylabel(ylabel, fontsize=7)
        ax1.set_title("k-means", fontsize=20)
        for k in range(self.k):
            ax1.plot(
                self.centroids[k][2], self.centroids[k][3], marker="*", color=(0, 0, 1)
            )
        color_list = np.array(self.km.color_list1)
        ax1.scatter(
            self.df["petal_length"],
            self.df["petal_width"],
            c=self.cluster,
            s=50,
            alpha=0.5,
        )
        ax2.scatter(
            self.df["petal_length"],
            self.df["petal_width"],
            c=self.km.color_list1,
            s=50,
            alpha=0.5,
        )
        self.scatter1 = FigureCanvasTkAgg(self.figure1, self.root)
        self.scatter1.draw()
        self.scatter1.get_tk_widget().grid(row=0, column=1, pady=0)


if __name__ == "__main__":
    gui = TkGUI()
    gui.makegui()
