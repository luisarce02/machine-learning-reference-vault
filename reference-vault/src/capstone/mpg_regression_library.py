import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

def load_and_prepare_data(file_path):
    """
    Remove unnecessary columns,
    and separate the features (X) from the target (y).
    """
    print("Cargando y preparando los datos...")
    data = pd.read_csv(file_path)
    data.drop(columns=['Unnamed: 0', 'car name'], inplace=True)
    target = data.pop('mpg')
    print(f"Datos cargados: {data.shape[0]} filas y {data.shape[1]} columnas (sin incluir el objetivo).")
    return data, target

def train_lasso_model(X, y, test_size=0.33, random_state=42):
    """
    Splits the data into training and testing, trains a Lasso model with cross-validation,
    and generates predictions.
    """
    print("\nDividiendo los datos en conjunto de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras.")
    print(f"Conjunto de prueba: {X_test.shape[0]} muestras.")

    print("\nEntrenando el modelo Lasso con validación cruzada (cv=5)...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', LassoCV(cv=5))
    ])
    pipeline.fit(X_train, y_train)
    print("Entrenamiento completado.")
    
    predictions = pipeline.predict(X_test)
    return y_test, predictions

def evaluate_model(y_true, y_pred):
    print("\nCalculando métricas de evaluación...")
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
    }
    return metrics

def plot_predictions(y_test, predictions, output_file="plot/predictions_vs_actual_library.png"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.6, color='blue', label="Predicciones")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label="Ideal")
    plt.xlabel("Valores Reales (MPG)")
    plt.ylabel("Predicciones (MPG)")
    plt.title("Comparación: Valores Reales vs Predicciones")
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    print(f"Gráfico guardado como {output_file}")

def main(file_path):
    print("Iniciando flujo del modelo...")
    
    X, y = load_and_prepare_data(file_path)
    
    y_test, predictions = train_lasso_model(X, y)
    
    metrics = evaluate_model(y_test, predictions)
    print("\nResultados de la evaluación del modelo Lasso:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    plot_predictions(y_test, predictions)
    
    print("\nProceso completado.")

file_path = 'data/auto_mpg.csv'

main(file_path)
