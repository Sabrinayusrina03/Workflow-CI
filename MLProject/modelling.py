import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# Load data clean
df = pd.read_csv("laptop_clean.csv")

X = df.drop(columns=["Price_euros"])
y = df["Price_euros"]

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(include="number").columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

model = Ridge()

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", model)
])

param_grid = {
    "model__alpha": [0.1, 1.0, 10.0]
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_tracking_uri("file:///tmp/mlruns")
mlflow.set_experiment("Laptop Price Prediction - Tuning")

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="r2"
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Manual logging (AMAN)
mlflow.log_param("alpha", grid.best_params_["model__alpha"])
mlflow.log_metric("MAE", mae)
mlflow.log_metric("R2", r2)

mlflow.sklearn.log_model(best_model, "model")

print("Best alpha:", grid.best_params_)
print("MAE:", mae)
print("R2:", r2)