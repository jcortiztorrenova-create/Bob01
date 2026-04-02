"""
Replay Buffer (Experience Replay) para DQN.

Almacena transiciones (s, a, r, s', done) y permite muestreos aleatorios
para romper la correlación temporal durante el entrenamiento.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple

import numpy as np


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    """Buffer circular de transiciones con muestreo uniforme."""

    def __init__(self, capacity: int) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Añade una transición al buffer."""
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Devuelve un mini-batch aleatorio como arrays de NumPy."""
        batch: List[Transition] = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def ready(self) -> bool:
        """True cuando hay suficientes muestras para el primer batch."""
        import config as cfg
        return len(self) >= cfg.BATCH_SIZE
