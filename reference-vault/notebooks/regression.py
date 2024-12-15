import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1.5, 3.1, 4.5, 6.0, 7.4])
model = LinearRegression().fit(X, y)
print("Regresión Lineal - Predicción para 6:", model.predict([[6]]))

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])
model = LogisticRegression().fit(X, y)
print("Regresión Logística - Predicción para 2.5:", model.predict([[2.5]]))

y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print("Error Cuadrático Medio (MSE):", mse)

accuracy = accuracy_score(y, model.predict(X))
print("Precisión del modelo:", accuracy)

# Red Neuronal Simple con PyTorch
# Definición de la red neuronal
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(1, 1)  # 1 entrada, 1 salida

    def forward(self, x):
        return self.linear(x)

# Preparación de datos
X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float).reshape(-1, 1)
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float).reshape(-1, 1)

# Conversión a tensores de PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Inicializar el modelo
model = SimpleNN()

# Definir el optimizador y la función de pérdida
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Entrenamiento de la red neuronal
epochs = 500
for epoch in range(epochs):
    model.train()

    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward pass y optimización
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Predicción con la red neuronal entrenada
model.eval()
with torch.no_grad():
    prediction = model(torch.tensor([[10.0]], dtype=torch.float32))
    print("Predicción con la red neuronal para 10:", prediction.item())
