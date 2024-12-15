import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


def load_and_prepare_data(file_path):
    """
    Remove unnecessary columns,
    and separate the features (X) from the target (y).
    """
    print("Cargando y preparando los datos...")
    data = pd.read_csv(file_path)
    data.drop(columns=["Unnamed: 0", "car name"], inplace=True)
    target = data.pop("mpg").values
    features = data.values
    print(
        f"Datos cargados: {data.shape[0]} filas y {data.shape[1]} columnas de características."
    )
    return features, target


def normalize_features(X):
    """
    Using the mean and standard deviation.
    Returns the normalized data along with the original mean and standard deviation.
    """
    print("\nNormalizando las características...")
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    print("Normalización completada.")
    return (X - mean) / std, mean, std


def compute_cost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors**2)
    return cost


def gradient_descent(X, y, theta, learning_rate, iterations):
    """
    Minimize the cost function.
    """
    print("\nIniciando el entrenamiento mediante descenso de gradiente...")
    m = len(y)
    cost_history = []
    for i in range(iterations):
        gradient = (1 / m) * (X.T @ (X @ theta - y))
        theta -= learning_rate * gradient
        cost_history.append(compute_cost(X, y, theta))
        if (i + 1) % (iterations // 10) == 0:
            print(f"Iteración {i + 1}/{iterations}: Costo = {cost_history[-1]:.4f}")
    print("Entrenamiento completado.")
    return theta, cost_history


def predict(X, theta):
    """
    Using the trained parameters (theta).
    """
    return X @ theta


# Cost evolution
def plot_cost_history(cost_history, output_file="plot/mpg_cost_history_manual.png"):
    print("\nGenerando gráfico de la evolución del costo...")
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(cost_history)), cost_history, label="Costo")
    plt.xlabel("Iteraciones")
    plt.ylabel("Costo (MSE)")
    plt.title("Evolución del costo durante el entrenamiento")
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    print(f"Gráfico guardado como {output_file}")


def plot_predictions_vs_actual(
    y_actual, y_predicted, output_file="plot/mpg_predictions_vs_actual_manual.png"
):
    print("\nGenerando gráfico de predicciones vs valores reales...")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, y_predicted, alpha=0.6, label="Predicciones")
    plt.plot(
        [y_actual.min(), y_actual.max()],
        [y_actual.min(), y_actual.max()],
        color="red",
        linestyle="--",
        label="Línea ideal",
    )
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")
    plt.title("Predicciones vs Valores Reales")
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    print(f"Gráfico guardado como {output_file}")


def train_and_evaluate_manual(file_path, learning_rate=0.01, iterations=1000):
    print("Iniciando el flujo del modelo manual...")

    features, target = load_and_prepare_data(file_path)

    features, mean, std = normalize_features(features)
    m, n = features.shape
    features = np.c_[np.ones(m), features]  # Add column of ones for bias term
    target = target.reshape(-1, 1)

    theta = np.zeros((n + 1, 1))

    # Train the model
    theta, cost_history = gradient_descent(
        features, target, theta, learning_rate, iterations
    )

    predictions = predict(features, theta)

    print("\nCalculando métricas de evaluación...")
    mae = mean_absolute_error(target, predictions)
    r2 = r2_score(target, predictions)
    print("Evaluación del modelo:")
    print(f"- Error Absoluto Medio (MAE): {mae:.4f}")
    print(f"- Coeficiente de Determinación (R2): {r2:.4f}")

    plot_cost_history(cost_history)
    plot_predictions_vs_actual(target, predictions)

    print("\nProceso completado.")
    return theta, cost_history, predictions, mean, std


file_path = "data/auto_mpg.csv"

theta, cost_history, predictions, mean, std = train_and_evaluate_manual(file_path)
