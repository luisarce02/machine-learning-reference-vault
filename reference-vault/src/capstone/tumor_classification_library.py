import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_and_clean_data(file_path):
    print("Cargando datos...")
    data = pd.read_csv(file_path)
    data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    print("Datos cargados y procesados.")
    return data


def preprocess_data(data):
    print("Dividiendo datos en entrenamiento y prueba...")
    X = data.drop(["diagnosis"], axis=1)
    y = data["diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print("Datos divididos.")
    return X_train, X_test, y_train, y_test


def standardize_data(X_train, X_test):
    """Standardize the data so that they have a mean of 0 and a standard deviation of 1."""
    print("Estandarizando datos...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Datos estandarizados.")
    return X_train_scaled, X_test_scaled


def train_logistic_regression(X_train, y_train):
    print("Entrenando modelo de regresión logística...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("Modelo entrenado.")
    return model


def plot_results(y_test, predictions):
    print("Graficando resultados...")
    plt.figure(figsize=(10, 6))

    # Index for data
    indices = np.arange(len(y_test))

    # Sort index by tags
    sorted_indices = np.argsort(y_test)

    # Real and predicted tags
    y_test_sorted = y_test.iloc[sorted_indices].values
    predictions_sorted = np.array(predictions)[sorted_indices]

    plt.scatter(indices, y_test_sorted, color="blue", label="Real", alpha=0.6)
    plt.scatter(
        indices,
        predictions_sorted,
        color="red",
        marker="x",
        label="Predicción",
        alpha=0.6,
    )

    plt.title("Resultados de Clasificación")
    plt.xlabel("Índice de Muestra Ordenada")
    plt.ylabel("Etiqueta")
    plt.legend()
    plt.savefig("../capstone/plot/tumor_classification_results_library.png")
    print(
        "Resultados graficados y guardados como 'tumor_classification_results_library.png'."
    )


def evaluate_model(model, X_test, y_test):
    print("Evaluando modelo...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Precisión del modelo: {accuracy:.2f}")
    plot_results(y_test, predictions)
    return accuracy


def main():
    file_path = "../capstone/data/breast_cancer.csv"
    data = load_and_clean_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)
    model = train_logistic_regression(X_train_scaled, y_train)
    accuracy = evaluate_model(model, X_test_scaled, y_test)
    print(f"Precisión del modelo: {accuracy:.2f}")
