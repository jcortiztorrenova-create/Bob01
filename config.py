"""
Configuración global del proyecto MARL Táctico.

Formalización POSG (Partially Observable Stochastic Game):
  - S:  espacio de estados (posiciones, munición, salud de todas las entidades)
  - A:  espacio de acciones conjuntas (producto cartesiano de acciones individuales)
  - T:  función de transición (determinista para movimiento, estocástica para disparo)
  - R:  función de recompensa por agente
  - O:  función de observación (radio limitado → observabilidad parcial)
  - γ:  factor de descuento
"""

# ── Entorno ─────────────────────────────────────────────────────────────────
GRID_SIZE = 10          # Cuadrícula NxN
NUM_ALLIES = 3          # Agentes aliados (aprendizaje)
NUM_ENEMIES = 3         # Agentes enemigos (IA por reglas)

# ── Vehículos ────────────────────────────────────────────────────────────────
VEHICLE_HEALTH = 3      # Puntos de vida iniciales
VEHICLE_AMMO = 10       # Munición inicial
WEAPON_RANGE = 3        # Rango del arma en casillas (distancia Manhattan)
HIT_PROBABILITY = 0.75  # Probabilidad base de impacto a distancia 1
HIT_DECAY = 0.10        # Reducción de probabilidad por cada casilla extra de distancia

# ── Observabilidad parcial ───────────────────────────────────────────────────
OBS_RADIUS = 5          # Radio de observación de cada agente (distancia Chebyshev)

# ── Espacio de acciones ──────────────────────────────────────────────────────
# 0-3: Moverse (ARRIBA, ABAJO, IZQUIERDA, DERECHA)
# 4:   Quedarse quieto
# 5-7: Disparar al enemigo 0, 1 o 2
NUM_ACTIONS = 5 + NUM_ENEMIES  # = 8

# ── Espacio de observación ───────────────────────────────────────────────────
# Propio: x, y, salud, munición                          → 4
# Aliados (NUM_ALLIES-1): rel_x, rel_y, salud, muni, vis → 5 * 2 = 10
# Enemigos (NUM_ENEMIES): rel_x, rel_y, salud, visible   → 4 * 3 = 12
OBS_SIZE = 4 + (NUM_ALLIES - 1) * 5 + NUM_ENEMIES * 4  # = 26

# ── Recompensas ──────────────────────────────────────────────────────────────
REWARD_STEP          = -0.05   # Penalización por paso (incentiva eficiencia)
REWARD_ENEMY_KILLED  = 10.0    # Eliminar un enemigo
REWARD_ALLY_KILLED   = -10.0   # Perder un aliado
REWARD_MISSION_WIN   = 50.0    # Eliminar todos los enemigos
REWARD_MISSION_LOSE  = -50.0   # Perder todos los aliados
REWARD_NO_AMMO       = -0.5    # Intentar disparar sin munición
REWARD_OUT_OF_RANGE  = -0.2    # Disparar fuera de rango

# ── Hiperparámetros DQN ──────────────────────────────────────────────────────
LEARNING_RATE   = 1e-3
GAMMA           = 0.99         # Factor de descuento
BATCH_SIZE      = 64
REPLAY_CAPACITY = 50_000
TARGET_UPDATE   = 200          # Pasos entre actualizaciones de la red objetivo
TAU             = 0.005        # Suavidad de actualización de red objetivo (soft update)

# Exploración epsilon-greedy
EPS_START = 1.0
EPS_END   = 0.05
EPS_DECAY = 0.995              # Multiplicador por episodio

# Capas ocultas de la red neuronal
HIDDEN_LAYERS = [128, 64]

# ── Entrenamiento ────────────────────────────────────────────────────────────
MAX_EPISODES     = 5_000
MAX_STEPS        = 200         # Pasos máximos por episodio
EVAL_INTERVAL    = 500         # Cada cuántos episodios evaluar
EVAL_EPISODES    = 20          # Episodios de evaluación (sin exploración)
SAVE_INTERVAL    = 1_000
CHECKPOINT_DIR   = "checkpoints"
LOG_DIR          = "logs"

# ── Reproducibilidad ─────────────────────────────────────────────────────────
SEED = 42
