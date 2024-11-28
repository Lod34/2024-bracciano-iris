from sklearn.datasets import load_iris # Caricamento del dataset iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Caricamento del dataset
iris = load_iris()
X = iris.data # Matrice delle feature
y = iris.target # Vettore delle classi

# Visualizzazione della distribuzione delle classi
plt.figure(figsize=(14, 8)) # La figura è la finestra in cui vengono mostrati i grafici

# Plot per i sepali
plt.subplot(1, 2, 1) # Crea una griglia 1x2 di grafici e seleziona la prima cella
for i in range(3):
    mask = y == i
    print(mask)
    # mask è un array di booleani che ha True per le righe in cui y è uguale a i
    plt.scatter(X[mask, 0], X[mask, 1], label=iris.target_names[i]) 
    # plot.scatter(x, y, label) crea un grafico a dispersione con x e y come coordinate e label come nome della classe
    # X[mask, 0] seleziona la lunghezza del sepalo per le righe in cui y è uguale a i
    # I punti non vengono plottati tutti insieme perché si vuole dare un colore diverso a ciascuna classe (i)

plt.xlabel('Lunghezza sepalo')
plt.ylabel('Larghezza sepalo')
plt.title('Distribuzione dei Sepali')
plt.legend()

# Plot per i petali
plt.subplot(1, 2, 2) # Seleziona la seconda cella
for i in range(3):
    mask = y == i
    plt.scatter(X[mask, 2], X[mask, 3], label=iris.target_names[i])
plt.xlabel('Lunghezza petalo')
plt.ylabel('Larghezza petalo')
plt.title('Distribuzione dei Petali')
plt.legend()

plt.show()