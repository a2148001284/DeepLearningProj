import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif  # or use f_classif, mutual_info_classif
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
import optuna
import scipy.io as sio
from choices import load_data

# Load data  记得修改下方的SelectKBest(f_classif, k=300) k的数值 为特征数量
X , y , k = load_data()

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define objective function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0)
    }

    clf = GradientBoostingClassifier(**params)
    pipeline = Pipeline([
        ('feature_selection', SelectKBest(f_classif, k=k)),  # Change to chi2 or other suitable method
        ('classification', clf)
    ])
    loo = LeaveOneOut()
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy', n_jobs=-1)  # Use parallel processing
    return np.mean(scores), np.std(scores)

# Create Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial)[0], n_trials=500, n_jobs=-1)  # Use parallel processing

# Get best parameters and performance
best_params = study.best_params
best_accuracy, best_std = objective(optuna.trial.FixedTrial(best_params))

print("Best Parameters: ", best_params)
print("Best Accuracy: {:.2f} ± {:.2f}".format(best_accuracy, best_std))
