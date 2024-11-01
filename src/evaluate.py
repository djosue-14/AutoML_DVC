import yaml
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, root_mean_squared_error, f1_score

# Cargar parámetros
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Cargar los datos preprocesados y el preprocesador que se guardaron en preprocessing.py
_, X_test, _, y_test = joblib.load('data/processed/dataset_processed.pkl')

# Cargar los modelos entrenados
model_names = ["regresion_logistica", "random_forest"]
results = {}

# Evaluar modelo a modelo
for model_name in model_names:
    model = joblib.load(f'models/{model_name}_model.pkl')
    y_pred = model.predict(X_test)
    
    # Calcular metricas
    accuracy = accuracy_score(y_test, y_pred)
    mse = root_mean_squared_error(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, output_dict=True)
    confusion = confusion_matrix(y_test, y_pred)

    results[model_name] = {
        "accuracy": accuracy,
        "mse": mse,
        "f1_score": f1,
        "classification_report": report,
        "confusion_matrix": confusion.tolist()
    }

pd.DataFrame(results).to_csv("results/evaluation_metrics.csv", index=False)
joblib.dump(results, 'models/evaluation_results.pkl')
