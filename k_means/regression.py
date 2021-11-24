import numpy as np
import pandas as pd

df = pd.read_csv("data/auto-mpg.csv")
X_data = df.loc[:,['weight','horsepower']].values
Y_data = df.loc[:,'mpg'].values

def preprocess(X_data):
    X_data_copy = X_data.copy()
    for i,data in enumerate(X_data_copy):
        if data[1]=='?':
            X_data = np.delete(X_data,i,0)
            X_data[i] = [data[0],int(data[1])]
            print(X_data.shape)
    return np.array(X_data)

def main():
    w = np.random(1)