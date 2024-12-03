import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import optuna
from sklearn.model_selection import cross_val_score, LeaveOneOut
import scipy.io as sio
from choices import load_data

# Load data  记得修改下方的SelectKBest(f_classif, k=300) k的数值 为特征数量
X , y , k = load_data()

# 数据变换：Z分数标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 定义目标函数
def objective(trial):
    params = {
        'C': trial.suggest_float('C', 0.1, 10.0, log=True),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'degree': trial.suggest_int('degree', 2, 5),  # 多项式核的度数
        'coef0': trial.suggest_float('coef0', 0.0, 1.0),  # 核函数中的独立项
        'shrinking': trial.suggest_categorical('shrinking', [True, False]),  # 是否使用启发式缩减
        'tol': trial.suggest_float('tol', 1e-5, 1e-1, log=True),  # 停止标准的容差
        'cache_size': trial.suggest_int('cache_size', 100, 1000)  # 核缓存的大小
    }

    clf = SVC(**params)
    pipeline = Pipeline([
        ('feature_selection', SelectKBest(f_classif, k=k)),
        ('classification', clf)
    ])
    loo = LeaveOneOut()

    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy', n_jobs=-1)  # 使用并行处理
    return np.mean(scores), np.std(scores)

# 创建Optuna研究对象
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial)[0], n_trials=1000, n_jobs=-1)  # 使用并行处理

# 获取最佳参数和性能
best_params = study.best_params
best_accuracy, best_std = objective(optuna.trial.FixedTrial(best_params))

print("最佳参数: ", best_params)
print("最佳准确率: {:.2f} ± {:.2f}".format(best_accuracy, best_std))
