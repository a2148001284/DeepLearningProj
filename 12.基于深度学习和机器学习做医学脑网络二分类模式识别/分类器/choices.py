import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import optuna
import scipy.io as sio

# 加载数据
def load_data():
    k = 80
    flag = 1
    if flag == 1:  # csv文件
        filename = "../data/灰质值.csv"
        data = np.loadtxt(filename, delimiter=',', dtype=np.float64)
        X = data[:, :-1]  # Features
        y = data[:, -1]  # Labels
        print(X.shape)  # (548, 90)  第一个为个数 第二个为特征值的个数
        print(y.shape)  # (548,) 一维的标记值矩阵
    elif flag == 2:  # feature.mat
        mat_file = sio.loadmat('../data/feature.mat')
        my_data = mat_file['feature']
        print(my_data.shape)
        X = my_data[0:300:1, :]
        y = my_data[300:301:1, :]
        y = y.T
        y = y.ravel()  # 处理为一维向量
        X = X.T
        print(X.shape)  # (558, 300)
        print(y.shape)  # (558,)
    elif flag == 3:
        mat_file = sio.loadmat('../data/FC90.mat')
        my_data = mat_file['FC90']
        print(my_data.shape)  # (4006, 558)
        X = my_data[0:4005:1, :]
        y = my_data[4005:4006:1, :]
        X = X.T
        y = y.T
        y = y.ravel()
        print(X.shape)  # (558, 4005)
        print(y.shape)  # (558,)
    elif flag == 4:
        mat_file = sio.loadmat('../data/FC116.mat')
        my_data = mat_file['FC116']
        print(my_data.shape)
        X = my_data[0:6670:1, :]
        y = my_data[6670:6671:1, :]
        X = X.T
        y = y.T
        y = y.ravel()
        print(X.shape)
        print(y.shape)
    return X , y , k
