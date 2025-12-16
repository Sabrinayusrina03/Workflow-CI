import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="../laptop_clean.csv")
args = parser.parse_args()

# Konfigurasi User
os.environ["MLFLOW_TRACKING_USERNAME"] = "Sabrinayusrina03" 

# Konfigurasi URI
mlflow.set_tracking_uri("https://dagshub.com/Sabrinayusrina03/eksperimen_SML_SabrinaYusrina.mlflow")

# Load data clean
df = pd.read_csv(args.data_path)

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

mlflow.set_experiment("Laptop Price Prediction - Tuning")

with mlflow.start_run():
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

    # Manual logging
    mlflow.log_param("alpha", grid.best_params_["model__alpha"])
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)

    mlflow.sklearn.log_model(best_model, "model")

    print("Best alpha:", grid.best_params_)
    print("MAE:", mae)
    print("R2:", r2)

    # Artefak 1: Residual Plot
    # Hitung Residual
    residuals = y_test - y_pred

    # Buat Plot
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, edgecolors=(0, 0, 0))
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')

    # Log Plot ke DagsHub
    mlflow.log_figure(fig, "Residual_Plot.png")

    plt.close(fig)

    #Artefak 2: Metrics Text File (.txt)
    metrics_file_path = "model_metrics_summary.txt"

    with open(metrics_file_path, 'w') as f:
        f.write(f"Model: {type(best_model).__name__}\n") 
        f.write("-" * 25 + "\n")
        f.write(f"MAE (Mean Absolute Error): {mean_absolute_error(y_test, y_pred)}\n")
        f.write(f"R2 Score: {r2_score(y_test, y_pred)}\n")

    # Log file teks ke DagsHub
    mlflow.log_artifact(metrics_file_path)

    os.remove(metrics_file_path)