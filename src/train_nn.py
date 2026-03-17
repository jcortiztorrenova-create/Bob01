from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Cargar dataset Iris
iris = load_iris()

print("Feature names:",iris.feature_names)
print("Class names:", iris.target_names)

print("\nPrimeros 5 datos:")
print(iris.data[:5])

print("\nClases de esos datos:")
print(iris.target[:5])

X = iris.data
y = iris.target

# 2. Separar entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Crear red neuronal pequeña
model = MLPClassifier(
    hidden_layer_sizes=(10,10),
    max_iter=200,
    verbose=True,
    random_state=42
)

# 4. Entrenar el modelo
model.fit(X_train, y_train)

# 5. Hacer predicciones
y_pred = model.predict(X_test)

# 6. Evaluar modelo
accuracy = accuracy_score(y_test, y_pred)

print("Precisión del modelo:", accuracy)

print("\nReporte de clasificación:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))