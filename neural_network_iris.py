import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =============================================================================
# METODOLOGIA PARA DETECTAR MAX ITERACIONES SOPORTADAS POR EL HARDWARE
# =============================================================================
# Se usa un enfoque de escalado exponencial (binary search):
#   1. Se ejecutan bloques de iteraciones crecientes (warm_up=False con partial_fit)
#   2. Se mide el tiempo por iteracion y el uso de memoria
#   3. Se define un presupuesto de tiempo maximo (TIME_BUDGET_SECONDS)
#   4. Se calcula cuantas iteraciones caben en ese presupuesto
#   5. Se verifica que la memoria disponible no se agote
# =============================================================================


def get_available_memory_mb():
    """Lee memoria disponible desde /proc/meminfo (Linux)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1024  # KB -> MB
    except FileNotFoundError:
        pass
    return None


def get_hardware_info():
    """Recopila informacion del hardware disponible."""
    info = {
        "cpus": os.cpu_count() or 1,
        "mem_available_mb": get_available_memory_mb(),
    }
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["mem_total_mb"] = int(line.split()[1]) / 1024
                    break
    except FileNotFoundError:
        info["mem_total_mb"] = None
    return info


def benchmark_iteration_speed(X_train, y_train, hidden_layers, n_samples=50):
    """
    Ejecuta n_samples iteraciones con partial_fit para medir
    el tiempo promedio por iteracion.
    """
    classes = np.unique(y_train)
    bench_model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=1,
        warm_start=True,
        random_state=42,
    )

    # Primera llamada inicializa pesos
    bench_model.partial_fit(X_train, y_train, classes=classes)

    times = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        bench_model.partial_fit(X_train, y_train)
        times.append(time.perf_counter() - t0)

    return {
        "mean_sec": np.mean(times),
        "std_sec": np.std(times),
        "min_sec": np.min(times),
        "max_sec": np.max(times),
        "total_samples": n_samples,
    }


def detect_max_iterations(X_train, y_train, hidden_layers,
                          time_budget_sec=300, memory_safety_pct=80):
    """
    Detecta el maximo de iteraciones que el hardware puede ejecutar
    dentro del presupuesto de tiempo y sin exceder el % de memoria.

    Parametros:
    -----------
    time_budget_sec : int
        Segundos maximos que se permite entrenar (default 5 min).
    memory_safety_pct : int
        % maximo de memoria total que se permite usar (default 80%).

    Retorna:
    --------
    dict con max_iter, benchmark, hardware_info y justificacion.
    """
    hw = get_hardware_info()
    bench = benchmark_iteration_speed(X_train, y_train, hidden_layers)

    # Calcular max iteraciones por tiempo
    max_by_time = int(time_budget_sec / bench["mean_sec"])

    # Calcular max iteraciones por memoria
    # MLPClassifier usa memoria constante (no crece con iteraciones),
    # pero verificamos que hay suficiente memoria disponible
    max_by_memory = float("inf")
    if hw["mem_available_mb"] is not None and hw["mem_total_mb"] is not None:
        mem_limit = hw["mem_total_mb"] * (memory_safety_pct / 100)
        mem_used = hw["mem_total_mb"] - hw["mem_available_mb"]
        if mem_used > mem_limit:
            max_by_memory = 0
        # Para MLP la memoria es constante, no limita iteraciones
        # Solo verificamos que hay memoria suficiente para operar

    max_iter = min(max_by_time, max_by_memory)

    # Tope practico: mas alla de cierto punto no mejora (convergencia)
    # Permitimos hasta el maximo calculado, el early_stopping se encargara
    practical_cap = 1_000_000
    max_iter = min(int(max_iter), practical_cap)

    return {
        "max_iter": max_iter,
        "time_budget_sec": time_budget_sec,
        "estimated_time_sec": max_iter * bench["mean_sec"],
        "benchmark": bench,
        "hardware": hw,
        "justification": (
            f"Con {bench['mean_sec']*1000:.3f} ms/iter promedio, "
            f"caben {max_iter:,} iteraciones en {time_budget_sec}s. "
            f"Hardware: {hw['cpus']} CPUs, "
            f"{hw['mem_total_mb']:.0f} MB RAM total, "
            f"{hw['mem_available_mb']:.0f} MB disponible."
        ),
    }


# =============================================================================
# IMPLEMENTACION
# =============================================================================

print("=" * 60)
print("DETECCION DE HARDWARE Y BENCHMARK")
print("=" * 60)

# 1 cargar dataset
iris = load_iris()
X = iris.data[:, 2:4]  # petal length, petal width
y = iris.target

# 2 separar entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3 detectar max iteraciones
HIDDEN_LAYERS = (10, 10)
TIME_BUDGET = 300  # 5 minutos maximo de entrenamiento

detection = detect_max_iterations(
    X_train, y_train, HIDDEN_LAYERS, time_budget_sec=TIME_BUDGET
)

print(f"\nHardware detectado:")
print(f"  CPUs:             {detection['hardware']['cpus']}")
print(f"  RAM total:        {detection['hardware']['mem_total_mb']:.0f} MB")
print(f"  RAM disponible:   {detection['hardware']['mem_available_mb']:.0f} MB")
print(f"\nBenchmark ({detection['benchmark']['total_samples']} iteraciones):")
print(f"  Tiempo/iter:      {detection['benchmark']['mean_sec']*1000:.3f} ms "
      f"(+/- {detection['benchmark']['std_sec']*1000:.3f} ms)")
print(f"  Min:              {detection['benchmark']['min_sec']*1000:.3f} ms")
print(f"  Max:              {detection['benchmark']['max_sec']*1000:.3f} ms")
print(f"\nResultado:")
print(f"  max_iter:         {detection['max_iter']:,}")
print(f"  Tiempo estimado:  {detection['estimated_time_sec']:.1f} s")
print(f"\nJustificacion: {detection['justification']}")

# 4 crear red neuronal con max_iter detectado + early_stopping
print(f"\n{'=' * 60}")
print(f"ENTRENAMIENTO (max_iter={detection['max_iter']:,}, early_stopping=True)")
print(f"{'=' * 60}\n")

model = MLPClassifier(
    hidden_layer_sizes=HIDDEN_LAYERS,
    max_iter=detection["max_iter"],
    early_stopping=True,       # detener si no mejora (evita iteraciones inutiles)
    n_iter_no_change=200,      # paciencia: 200 iteraciones sin mejora
    tol=1e-6,                  # tolerancia muy baja para exprimir convergencia
    validation_fraction=0.15,  # 15% de train para validacion interna
    verbose=True,
    random_state=42,
)

t_start = time.perf_counter()
model.fit(X_train, y_train)
t_train = time.perf_counter() - t_start

print(f"\nEntrenamiento completado en {t_train:.2f} s")
print(f"Iteraciones reales: {model.n_iter_}")
print(f"Loss final: {model.loss_:.6f}")
print(f"Score en test: {model.score(X_test, y_test):.4f}")

# 5 crear grid para visualizar fronteras
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200),
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid)
Z = Z.reshape(xx.shape)

# 6 dibujar frontera de decision y matriz de confusion
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].contourf(xx, yy, Z, alpha=0.3)
axes[0].scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
axes[0].set_xlabel("petal length")
axes[0].set_ylabel("petal width")
axes[0].set_title(
    f"Neural Network Decision Boundary\n"
    f"(max_iter={detection['max_iter']:,}, "
    f"real_iter={model.n_iter_}, "
    f"score={model.score(X_test, y_test):.2%})"
)

# 7 matriz de confusion
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(ax=axes[1], colorbar=False)
axes[1].set_title("Confusion Matrix (test set)")

plt.tight_layout()
plt.show()
