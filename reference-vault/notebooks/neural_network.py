import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float).reshape(-1, 1)
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float).reshape(-1, 1)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

model = SimpleNN()

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

epochs = 500
for epoch in range(epochs):
    model.train()

    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

model.eval()
with torch.no_grad():
    prediction = model(torch.tensor([[10.0]], dtype=torch.float32))
    print("Predicción con la red neuronal para 10:", prediction.item())
