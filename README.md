# Neural Network - Iris Classification

Clasificación del dataset Iris usando una red neuronal (MLPClassifier) con visualización de fronteras de decisión.

## Requisitos

```bash
pip install numpy matplotlib scikit-learn
```

## Código

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# 1 cargar dataset
iris = load_iris()
# usar solo dos variables para poder dibujar
X = iris.data[:, 2:4]  # petal length, petal width
y = iris.target

# 2 separar entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3 crear red neuronal
model = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    max_iter=1000,
    verbose=True,
    random_state=42
)

# 4 entrenar
model.fit(X_train, y_train)

# 5 crear grid para visualizar fronteras
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid)
Z = Z.reshape(xx.shape)

# 6 dibujar frontera de decisión
plt.contourf(xx, yy, Z, alpha=0.3)
# dibujar puntos reales
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.title("Neural Network Decision Boundary")
plt.show()
```

## Descripción

- **Dataset**: Iris (solo petal length y petal width)
- **Modelo**: MLPClassifier con 2 capas ocultas de 10 neuronas cada una
- **Visualización**: Frontera de decisión con `contourf` y puntos reales del dataset
