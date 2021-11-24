import numpy as np
from numpy import linalg as LA
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Process Regression')
parser.add_argument('-f','--features',action="append",help='select feture')

class Regression:
    def __init__(self,features):
        df = pd.read_csv("../data/auto-mpg.csv")
        self.features = list(map(str,features))
        print("selected fetures:",self.features)
        self.X_data = df.loc[:,self.features].values
        self.Y_data = df.loc[:,'mpg'].values
        if 'horsepower' in self.features:
            idx = self.features.index("horsepower")
            self.X_data,self.Y_data = self.preprocess(self.X_data,self.Y_data,idx)
            print(self.X_data.shape)
    def preprocess(self,X_data,Y_data,idx):
        new_data = []
        
        new_Y = []
        for data,t in zip(X_data,Y_data):
            if data[idx]=='?':
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
        return np.array(new_data),np.array(new_Y)

    def predict(self,X_data,Y_data):

        num_data,n_feature = X_data.shape
        X_data = X_data.astype("float")
        b = np.ones((num_data,1))
        X_data = np.hstack((X_data,b))

        # 重みとバイアスの初期化
        W = np.random.randn(n_feature+1)
        pred1 = np.dot(X_data,W.T)
        loss1 = LA.norm(Y_data-pred1)/2

        # Wを更新 ムーア・ペローンズの疑似逆行列
        A_T =  LA.inv(np.dot(X_data.T,X_data))
        W = np.dot(np.dot(A_T,X_data.T),Y_data)
        pred2 = np.dot(X_data,W.T)
        loss2 = loss = LA.norm(Y_data-pred2)/2
        print("before loss:",loss1)
        print("after loss:",loss2)

if __name__=="__main__":
    args = parser.parse_args()
    features = args.features
    model = Regression(features)
    model.predict(model.X_data,model.Y_data)