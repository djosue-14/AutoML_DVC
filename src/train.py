import yaml
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Cargar parámetros
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Cargar los datos preprocesados
X_train, X_test, y_train, y_test = joblib.load('data/processed/dataset_processed.pkl')

# Definir modelos y parámetros
models = {
    "regresion_logistica": LogisticRegression(),
    "random_forest": RandomForestClassifier(),
    #"xgboost": XGBClassifier()
}

param_grids = {
    "regresion_logistica": {
        "solver": params["models"]["regresion_logistica"]["solver"],
        "C": params["models"]["regresion_logistica"]["C"],
        "max_iter": params["models"]["regresion_logistica"]["max_iter"]
    },
    "random_forest": {
        "n_estimators": params["models"]["random_forest"]["n_estimators"],
        "max_depth": params["models"]["random_forest"]["max_depth"],
        "min_samples_split": params["models"]["random_forest"]["min_samples_split"]
    },
    #"xgboost": {
    #    "eta": params["models"]["xgboost"]["eta"],
    #    "max_depth": params["models"]["xgboost"]["max_depth"],
    #    "subsample": params["models"]["xgboost"]["subsample"],
    #    "colsample_bytree": params["models"]["xgboost"]["colsample_bytree"]
    #}
}

# Búsqueda de hiperparámetros
best_models = {}
feature_importances = {}
for model_name, model in models.items():
    param_grid = param_grids[model_name]
    search = RandomizedSearchCV(
        model, param_distributions=param_grid,
        n_iter=params["hyperparameter_search"]["n_iter"],
        scoring=params["hyperparameter_search"]["scoring"],
        n_jobs=params["hyperparameter_search"]["n_jobs"],
        cv=params["hyperparameter_search"]["cv_folds"],
        random_state=params["data"]["split"]["random_state"]
    )
    search.fit(X_train, y_train)
    best_models[model_name] = search.best_estimator_
    
    # Guardar el mejor modelo
    joblib.dump(search.best_estimator_, f'models/{model_name}_model.pkl')

    # Obtener e identificar características importantes si están disponibles
    if hasattr(search.best_estimator_, 'feature_importances_'):
        feature_importances[model_name] = search.best_estimator_.feature_importances_

# Exportar características importantes
if feature_importances:
    pd.DataFrame(feature_importances).to_csv("results/feature_importances.csv", index=False)
