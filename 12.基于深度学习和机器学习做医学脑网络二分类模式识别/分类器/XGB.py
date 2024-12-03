import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression,f_classif
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import optuna
from choices import load_data

X , y , k = load_data()

# 数据变换：Z分数标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 定义目标函数
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 600),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 4),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.6),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
    }

    clf = xgb.XGBClassifier(tree_method='hist', **params)  # 使用并行处理
    pipeline = Pipeline([
        ('feature_selection', SelectKBest(f_classif, k=k)),
        ('classification', clf)
    ])
    loo = LeaveOneOut()
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy', n_jobs=-1)  # 使用并行处理
    return np.mean(scores), np.std(scores)

# 创建Optuna研究对象
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial)[0], n_trials=300, n_jobs=-1)  # 使用并行处理

# 获取最佳参数和性能
best_params = study.best_params
best_accuracy, best_std = objective(optuna.trial.FixedTrial(best_params))

print("最佳参数: ", best_params)
print("最佳准确率: {:.2f} ± {:.2f}".format(best_accuracy, best_std))
