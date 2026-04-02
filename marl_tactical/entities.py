"""
Entidades del campo de batalla.

Cada Vehicle representa un vehículo táctico con:
  - Posición (x, y) en la cuadrícula
  - Salud (health): al llegar a 0 el vehículo queda destruido
  - Munición (ammo): disminuye con cada disparo; sin munición no puede disparar
  - Rango de arma (weapon_range): distancia máxima de disparo (Manhattan)
  - Probabilidad de impacto (hit_prob): probabilidad base a distancia 1
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


class Team(Enum):
    ALLY  = "ally"
    ENEMY = "enemy"


@dataclass
class Vehicle:
    """Vehículo táctico en el entorno POSG."""

    id:           int
    team:         Team
    x:            int
    y:            int
    health:       int   = 3
    max_health:   int   = field(init=False)
    ammo:         int   = 10
    max_ammo:     int   = field(init=False)
    weapon_range: int   = 3    # casillas (distancia Manhattan)
    hit_prob:     float = 0.75 # probabilidad base a distancia 1

    def __post_init__(self) -> None:
        self.max_health = self.health
        self.max_ammo   = self.ammo

    # ── Propiedades ──────────────────────────────────────────────────────────

    @property
    def alive(self) -> bool:
        return self.health > 0

    @property
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)

    @property
    def has_ammo(self) -> bool:
        return self.ammo > 0

    # ── Acciones ──────────────────────────────────────────────────────────────

    def move(self, dx: int, dy: int, grid_size: int) -> bool:
        """
        Desplaza el vehículo en la dirección (dx, dy).
        Retorna True si el movimiento es válido (dentro de la cuadrícula).
        """
        nx, ny = self.x + dx, self.y + dy
        if 0 <= nx < grid_size and 0 <= ny < grid_size:
            self.x, self.y = nx, ny
            return True
        return False

    def distance_to(self, other: "Vehicle") -> int:
        """Distancia Manhattan al objetivo."""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def in_range(self, target: "Vehicle") -> bool:
        """Comprueba si el objetivo está dentro del rango del arma."""
        return self.alive and target.alive and self.distance_to(target) <= self.weapon_range

    def shoot(self, target: "Vehicle", rng) -> Tuple[bool, bool]:
        """
        Intenta disparar al objetivo.

        Retorna (shot_fired: bool, hit: bool):
          - shot_fired=False si no hay munición o el objetivo está fuera de rango.
          - hit=True  si el disparo impacta (la salud del objetivo decrece en 1).

        La probabilidad de impacto decrece linealmente con la distancia:
            p_hit = max(0, hit_prob - hit_decay * (dist - 1))
        donde hit_decay viene de config.
        """
        import config as cfg

        if not self.has_ammo:
            return False, False
        if not self.in_range(target):
            return False, False

        self.ammo -= 1
        dist     = self.distance_to(target)
        p_hit    = max(0.0, self.hit_prob - cfg.HIT_DECAY * (dist - 1))
        hit      = rng.random() < p_hit

        if hit:
            target.health = max(0, target.health - 1)

        return True, hit

    # ── Serialización ─────────────────────────────────────────────────────────

    def obs_vector(self, grid_size: int):
        """
        Vector de observación normalizado para este vehículo:
        [x/N, y/N, health/max_health, ammo/max_ammo]
        """
        return [
            self.x / grid_size,
            self.y / grid_size,
            self.health / self.max_health,
            self.ammo   / self.max_ammo,
        ]

    def __repr__(self) -> str:
        symbol = "A" if self.team == Team.ALLY else "E"
        status = "vivo" if self.alive else "destruido"
        return (
            f"{symbol}{self.id}(pos=({self.x},{self.y}), "
            f"hp={self.health}/{self.max_health}, "
            f"ammo={self.ammo}/{self.max_ammo}, {status})"
        )
