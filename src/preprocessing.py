import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import yaml

# Cargar parámetros
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

data = pd.read_csv(params["data"]["path"])

# Preprocesamiento
target = params["preprocessing"]["target"]
features = params["preprocessing"]["features"]

X = data[features]
y = data[target]

numeric_features = [col for col in features if col in X.select_dtypes(include=['int64', 'float64']).columns]
categorical_features = [col for col in features if col in X.select_dtypes(include=['object']).columns]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=params["data"]["split"]["train_size"],
    random_state=params["data"]["split"]["random_state"]
)

# Crear el pipeline de preprocesamiento
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy=params["preprocessing"]["numeric"]["imputation_strategy"])),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy=params["preprocessing"]["categorical"]["imputation_strategy"])),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Transformar datos
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Guardar los datos preprocesados y el preprocesador
joblib.dump((X_train, X_test, y_train, y_test), 'data/processed/dataset_processed.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')
