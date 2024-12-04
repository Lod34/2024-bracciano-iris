from sklearn.datasets import load_iris # Caricamento del dataset iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import random

#}
#------------INIZIALIZZAZIONE----------------
#{

# Caricamento del dataset
iris = load_iris() # Carica il dataset iris
X = iris.data # Matrice delle feature
y = iris.target # Vettore delle classi

# Visualizzazione della distribuzione delle classi
plt.figure(figsize=(14, 8)) # La figura è la finestra in cui vengono mostrati i grafici

# Plot per i sepali
plt.subplot(1, 2, 1) # Crea una griglia 1x2 di grafici e seleziona la prima cella
for i in range(3):
    mask = y == i # mask = (y == i) è una maschera booleana
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
colors = ['black', 'blue', 'yellow']
for i in range(3):
    mask = y == i
    plt.scatter(X[mask, 2], X[mask, 3], label=iris.target_names[i], color=colors[i])
plt.xlabel('Lunghezza petalo')
plt.ylabel('Larghezza petalo')
plt.title('Distribuzione dei Petali')
plt.legend()

#plt.show()

#------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=99)
# train_test_split divide il dataset in due parti, una per il training e una per il test
# test_size corrisponde a "Ratio of training to test data" su playground.tensorflow.org
# random_state è il seed per la generazione di numeri casuali, sarebbe il tasto 'generate' su playground.tensorflow.org
#------------------------------------------------------------
# Perché se si aumenta il test_size l'errore si avvicina di più a zero? Perché si sta allenando il modello su più dati
# Allora cosa cambia da 'batch size' a 'test size'? Il batch size è il numero di esempi che vengono passati al modello in una volta sola

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # fit_transform calcola la media e la deviazione standard e normalizza i dati
print (X_train_scaled)
print(f"minimo: {min(X_train_scaled[:,0])}", 
      f"massimo: {max(X_train_scaled[:,0])}",
      f"media: {np.mean(X_train_scaled[:,0])}", # la media è 0 perché i dati sono stati normalizzati: la media è 0 e la deviazione standard è 1
      sep = "\n")

X_test_scaled = scaler.transform(X_test) # trasforma i dati di test in base ai parametri calcolati con i dati di training


#}
#------------CREAZIONE DEL MULTI LAYER PERCEPTRON----------------
#{

mlp = MLPClassifier(
    hidden_layer_sizes=(4,2), 
    activation='tanh',
    random_state=1,
    max_iter=5000, # le iterazioni si distinguono dalle epoche perché le epoche sono il numero di volte che il modello vede tutti i dati
    # sotto 100 da un warning: è come se su tensorflow.playgroung.org avvii la simulazione e subito dopo la blocchi; la rete non fa in tempo ad ottimizzarla
)

mlp.fit(X_train_scaled, y_train) # addestra il modello con i dati di training normalizzati
y_predict = mlp.predict(X_test_scaled) # predice le classi dei dati di test

accuracy = np.mean(y_predict == y_test) # calcola l'accuratezza del modello
print(f"Accuratezza: {accuracy:.2f}") # .2f indica che si vogliono due cifre decimali (f sta per float)

print(f"Test loss: {mlp.loss_}")
print(f"Numero di iterazioni: {mlp.n_iter_}")

#}
#------------VISUALIZZAZIONE DEI RISULTATI----------------
#{

nuovo_iris = [5.0, 3.5, 1.5, 0.2] # dati di un fiore di iris
nuovo_iris_scaled = scaler.transform([nuovo_iris]) # normalizza i dati del fiore. va messo tra parentesi quadre perché scaler.transform vuole una matrice perché è stato addestrato con una matrice
# Giustamente, chi è che crea una rete neurale per un solo dato? Si crea una rete neurale per un dataset.
# Noi, in questo caso, analizziamo un solo fiore
previsione_iris = mlp.predict(nuovo_iris_scaled) # predice la classe del fiore
print(f"Previsione del fiore: {iris.target_names[previsione_iris[0]]}")
#------------------------------------------------------------
# Nelle immagini, funziona alla stessa maniera, solo che si passano i pixel: si passano i pixel di un'immagine e si predice cosa c'è nell'immagine
