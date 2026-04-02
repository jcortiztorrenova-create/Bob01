"""
IA enemiga basada en reglas para el entorno MARL táctico.

Estrategia:
  1. Si hay aliados en rango y tiene munición → disparar al aliado más cercano.
  2. Si no puede disparar → avanzar hacia el aliado vivo más cercano.
  3. Si no hay aliados vivos → quedarse quieto.

Esta IA proporciona oponentes coherentes pero predecibles,
lo que obliga a los agentes MARL a aprender maniobras básicas
de fuego y movimiento para superarla.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from marl_tactical.environment import TacticalEnv


# Acciones: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY, 5/6/7=SHOOT ally[0/1/2]
_STAY = 4


def _move_towards(ex: int, ey: int, tx: int, ty: int) -> int:
    """Devuelve la acción de movimiento que acerca (ex,ey) a (tx,ty)."""
    dx = tx - ex
    dy = ty - ey
    if abs(dx) >= abs(dy):
        return 3 if dx > 0 else 2   # RIGHT / LEFT
    else:
        return 1 if dy > 0 else 0   # DOWN  / UP


class RuleBasedEnemyAI:
    """Callable que devuelve acciones para todos los enemigos vivos."""

    def __call__(self, env: "TacticalEnv") -> List[int]:
        """
        Calcula las acciones de los enemigos dado el estado actual del entorno.

        Retorna una lista de longitud NUM_ENEMIES (acción para cada enemigo,
        incluidos los destruidos, que se ignorarán en el entorno).
        """
        actions = []
        alive_allies = env.alive_allies

        for enemy in env.enemies:
            if not enemy.alive:
                actions.append(_STAY)
                continue

            if not alive_allies:
                actions.append(_STAY)
                continue

            # Encontrar al aliado vivo más cercano
            nearest_ally = min(
                alive_allies,
                key=lambda a: enemy.distance_to(a),
            )

            # ¿Puede disparar al aliado más cercano?
            if enemy.in_range(nearest_ally) and enemy.has_ammo:
                ally_action_idx = 5 + nearest_ally.id
                actions.append(ally_action_idx)
            else:
                # Avanzar hacia el aliado más cercano
                move_action = _move_towards(
                    enemy.x, enemy.y, nearest_ally.x, nearest_ally.y
                )
                actions.append(move_action)

        return actions
