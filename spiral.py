import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# Genera dati a spirale in coordinate polari
def generate_spiral_data(n_points, noise=0.1):
    r = np.linspace(0, 1, n_points)
    theta = np.linspace(0, 4 * np.pi, n_points)
    x = r * np.cos(theta) + noise * np.random.randn(n_points)
    y = r * np.sin(theta) + noise * np.random.randn(n_points)
    return np.vstack((x, y)).T

# Genera i dati
n_points = 1000
data = generate_spiral_data(n_points)
X = data[:, 0].reshape(-1, 1)
y = data[:, 1]

# Crea e addestra la rete neurale
mlp = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
mlp.fit(X, y)

# Predice i valori
y_pred = mlp.predict(X)

# Visualizza i risultati
plt.figure(figsize=(10, 5))
plt.scatter(data[:, 0], data[:, 1], label='Dati originali')
plt.scatter(data[:, 0], y_pred, label='Predizioni', alpha=0.5)
plt.legend()
plt.title('Rete Neurale per la Spirale')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()