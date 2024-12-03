import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import optuna
import scipy.io as sio
from choices import load_data

X , y , k = load_data()

# 数据变换：Z分数标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 定义目标函数
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }

    clf = RandomForestClassifier(**params)
    pipeline = Pipeline([
        ('feature_selection', SelectKBest(f_classif, k=k)),
        ('classification', clf)
    ])
    loo = LeaveOneOut()
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy', n_jobs=-1)  # Use parallel processing
    return np.mean(scores), np.std(scores)


# Create Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial)[0], n_trials=1000, n_jobs=-1)  # Use parallel processing

# Get best parameters and performance
best_params = study.best_params
best_accuracy, best_std = objective(optuna.trial.FixedTrial(best_params))

print("Best Parameters: ", best_params)
print("Best Accuracy: {:.2f} ± {:.2f}".format(best_accuracy, best_std))

