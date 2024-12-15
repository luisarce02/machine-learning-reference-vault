import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, weights):
    m = len(y)
    predictions = sigmoid(np.dot(X, weights))
    cost = (
        -1
        / m
        * (np.dot(y, np.log(predictions)) + np.dot((1 - y), np.log(1 - predictions)))
    )
    return cost


def gradient_descent(X, y, weights, learning_rate, iterations):
    """Optimize weights using gradient descent."""
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (predictions - y)) / m
        weights -= learning_rate * gradient

        cost = compute_cost(X, y, weights)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost:.4f}")

    return weights, cost_history


def predict(X, weights):
    """Binary predictions from probabilities."""
    probabilities = sigmoid(np.dot(X, weights))
    return [1 if prob >= 0.5 else 0 for prob in probabilities]


def plot_cost_convergence(cost_history, iterations):
    plt.figure(figsize=(8, 5))
    plt.plot(range(iterations), cost_history, color="blue")
    plt.title("Convergencia del Costo")
    plt.xlabel("Iteraciones")
    plt.ylabel("Costo")
    plt.savefig("../capstone/plot/tumor_cost_convergence_manual.png")
    print(
        "Gráfico de convergencia del costo guardado como 'tumor_cost_convergence_manual.png'."
    )


def plot_results(y_test, predictions):
    """Graphing the results comparing predictions against actual values."""
    print("Graficando resultados de clasificación...")
    plt.figure(figsize=(10, 6))

    # Index for data
    indices = np.arange(len(y_test))

    # Sort index by actual labels
    sorted_indices = np.argsort(y_test)

    # Real tags and predictions
    y_test_sorted = y_test[sorted_indices]
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
    plt.savefig("../capstone/plot/tumor_classification_results_manual.png")
    print("Resultados guardados como 'tumor_classification_results_manual.png'.")


def load_and_prepare_data(file_path):
    print("Cargando y procesando datos...")
    data = pd.read_csv(file_path)
    data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

    features = data.drop("diagnosis", axis=1).values
    target = data["diagnosis"].values

    # Normalize features
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    # Add bias
    features = np.c_[np.ones((features.shape[0], 1)), features]

    print("Datos preparados.")
    return features, target


def split_data(features, target, train_ratio=0.7):
    print("Dividiendo datos en entrenamiento y prueba...")
    train_size = int(train_ratio * len(features))
    X_train, X_test = features[:train_size], features[train_size:]
    y_train, y_test = target[:train_size], target[train_size:]
    print("Datos divididos.")
    return X_train, X_test, y_train, y_test


def main():
    file_path = "../capstone/data/breast_cancer.csv"
    features, target = load_and_prepare_data(file_path)

    X_train, X_test, y_train, y_test = split_data(features, target)

    # Initialize weights
    weights = np.zeros(X_train.shape[1])

    # Train model
    learning_rate = 0.01
    iterations = 1000
    weights, cost_history = gradient_descent(
        X_train, y_train, weights, learning_rate, iterations
    )

    plot_cost_convergence(cost_history, iterations)

    predictions = predict(X_test, weights)

    accuracy = np.mean(predictions == y_test)
    print(f"Precisión del modelo: {accuracy:.2f}")

    plot_results(y_test, predictions)

