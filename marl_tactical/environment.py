"""
Entorno POSG (Partially Observable Stochastic Game) táctico.

Formalización:
  S  = {(pos_i, hp_i, ammo_i) | i ∈ Aliados ∪ Enemigos}
  A  = A_1 × A_2 × ... × A_n  (acciones conjuntas)
  T  = P(s' | s, a)  — determinista para movimiento, estocástica para disparo
  R_i = f(evento)    — recompensa individual por agente aliado
  Ω_i = obs dentro del radio OBS_RADIUS para el agente i
  O_i = P(o_i | s)

Índices de acción por agente:
  0 → MOVE_UP     (dy=-1)
  1 → MOVE_DOWN   (dy=+1)
  2 → MOVE_LEFT   (dx=-1)
  3 → MOVE_RIGHT  (dx=+1)
  4 → STAY
  5 → SHOOT enemy[0]
  6 → SHOOT enemy[1]
  7 → SHOOT enemy[2]
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

import config as cfg
from marl_tactical.entities import Team, Vehicle


# Deltas de movimiento: ARRIBA, ABAJO, IZQUIERDA, DERECHA, QUIETO
_MOVE_DELTAS = [
    (0, -1),   # 0: MOVE_UP
    (0,  1),   # 1: MOVE_DOWN
    (-1, 0),   # 2: MOVE_LEFT
    ( 1, 0),   # 3: MOVE_RIGHT
    ( 0, 0),   # 4: STAY
]


class TacticalEnv:
    """
    Entorno táctico multiagente en cuadrícula NxN.

    Aliados controlados por agentes MARL (acciones recibidas en step()).
    Enemigos controlados por IA basada en reglas (enemy_ai).
    """

    def __init__(self, enemy_ai=None, seed: Optional[int] = None) -> None:
        self.grid_size = cfg.GRID_SIZE
        self.rng       = np.random.default_rng(seed)
        self.enemy_ai  = enemy_ai   # callable(env) → List[int] de acciones enemigas

        self.allies:  List[Vehicle] = []
        self.enemies: List[Vehicle] = []
        self.step_count = 0
        self.done = False

        # Estadísticas del episodio
        self.stats = {"enemy_kills": 0, "ally_losses": 0, "shots_fired": 0, "hits": 0}

    # ── Inicialización ────────────────────────────────────────────────────────

    def reset(self) -> List[np.ndarray]:
        """
        Reinicia el entorno y devuelve observaciones iniciales por agente aliado.
        Las posiciones iniciales se generan aleatoriamente evitando solapamientos.
        """
        self.step_count = 0
        self.done       = False
        self.stats      = {"enemy_kills": 0, "ally_losses": 0, "shots_fired": 0, "hits": 0}

        occupied: set = set()

        def random_pos() -> Tuple[int, int]:
            while True:
                pos = (
                    int(self.rng.integers(0, self.grid_size)),
                    int(self.rng.integers(0, self.grid_size)),
                )
                if pos not in occupied:
                    occupied.add(pos)
                    return pos

        # Aliados en la mitad izquierda, enemigos en la mitad derecha
        self.allies = []
        for i in range(cfg.NUM_ALLIES):
            x = int(self.rng.integers(0, self.grid_size // 2))
            y = int(self.rng.integers(0, self.grid_size))
            while (x, y) in occupied:
                x = int(self.rng.integers(0, self.grid_size // 2))
                y = int(self.rng.integers(0, self.grid_size))
            occupied.add((x, y))
            self.allies.append(Vehicle(
                id=i, team=Team.ALLY, x=x, y=y,
                health=cfg.VEHICLE_HEALTH, ammo=cfg.VEHICLE_AMMO,
                weapon_range=cfg.WEAPON_RANGE, hit_prob=cfg.HIT_PROBABILITY,
            ))

        self.enemies = []
        for i in range(cfg.NUM_ENEMIES):
            x = int(self.rng.integers(self.grid_size // 2, self.grid_size))
            y = int(self.rng.integers(0, self.grid_size))
            while (x, y) in occupied:
                x = int(self.rng.integers(self.grid_size // 2, self.grid_size))
                y = int(self.rng.integers(0, self.grid_size))
            occupied.add((x, y))
            self.enemies.append(Vehicle(
                id=i, team=Team.ENEMY, x=x, y=y,
                health=cfg.VEHICLE_HEALTH, ammo=cfg.VEHICLE_AMMO,
                weapon_range=cfg.WEAPON_RANGE, hit_prob=cfg.HIT_PROBABILITY,
            ))

        return self._get_observations()

    # ── Paso del entorno ──────────────────────────────────────────────────────

    def step(
        self,
        ally_actions: List[int],
    ) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        """
        Ejecuta un paso del entorno.

        Args:
            ally_actions: lista de longitud NUM_ALLIES con el índice de acción de cada aliado.

        Returns:
            observations: observaciones de cada aliado tras el paso.
            rewards:      recompensa de cada aliado.
            done:         True si el episodio terminó.
            info:         información de diagnóstico.
        """
        assert not self.done, "El episodio ya terminó. Llama a reset()."
        assert len(ally_actions) == cfg.NUM_ALLIES

        self.step_count += 1
        rewards = [cfg.REWARD_STEP] * cfg.NUM_ALLIES   # penalización por paso

        # 1. Ejecutar acciones de aliados ──────────────────────────────────────
        for agent_idx, action in enumerate(ally_actions):
            agent = self.allies[agent_idx]
            if not agent.alive:
                continue

            if action < 5:                          # movimiento o quieto
                dx, dy = _MOVE_DELTAS[action]
                agent.move(dx, dy, self.grid_size)
            else:                                   # disparo
                enemy_idx = action - 5
                if enemy_idx < len(self.enemies):
                    target = self.enemies[enemy_idx]
                    r_shot, r_hit = self._resolve_shot(agent, target, agent_idx, rewards)

        # 2. Eliminar enemigos destruidos ─────────────────────────────────────
        newly_killed = [e for e in self.enemies if not e.alive]
        for _ in newly_killed:
            self.stats["enemy_kills"] += 1
            for i in range(cfg.NUM_ALLIES):
                if self.allies[i].alive:
                    rewards[i] += cfg.REWARD_ENEMY_KILLED

        # 3. Ejecutar acciones de enemigos (IA por reglas) ─────────────────────
        if self.enemy_ai is not None:
            enemy_actions = self.enemy_ai(self)
            for e_idx, e_action in enumerate(enemy_actions):
                enemy = self.enemies[e_idx]
                if not enemy.alive:
                    continue
                self._execute_enemy_action(enemy, e_action, rewards)

        # 4. Comprobar bajas de aliados ────────────────────────────────────────
        for i, ally in enumerate(self.allies):
            if not ally.alive:
                # Solo contabilizar si acaba de morir (hp == 0 y no estaba antes)
                pass  # las bajas se detectan tras el disparo enemigo abajo

        # Detectar aliados recién eliminados
        for e_idx, enemy in enumerate(self.enemies):
            pass  # ya procesado arriba

        # 5. Condición de fin de episodio ──────────────────────────────────────
        alive_enemies = [e for e in self.enemies if e.alive]
        alive_allies  = [a for a in self.allies  if a.alive]

        if not alive_enemies:                        # victoria
            for i in range(cfg.NUM_ALLIES):
                rewards[i] += cfg.REWARD_MISSION_WIN
            self.done = True

        elif not alive_allies:                       # derrota
            for i in range(cfg.NUM_ALLIES):
                rewards[i] += cfg.REWARD_MISSION_LOSE
            self.done = True

        elif self.step_count >= cfg.MAX_STEPS:       # tiempo agotado
            self.done = True

        obs  = self._get_observations()
        info = {
            "step":        self.step_count,
            "alive_allies":  len(alive_allies),
            "alive_enemies": len(alive_enemies),
            **self.stats,
        }
        return obs, rewards, self.done, info

    # ── Mecánicas de disparo ──────────────────────────────────────────────────

    def _resolve_shot(
        self,
        shooter: Vehicle,
        target: Vehicle,
        agent_idx: int,
        rewards: List[float],
    ) -> Tuple[bool, bool]:
        """Resuelve un disparo del agente aliado. Modifica rewards in-place."""
        if not target.alive:
            return False, False

        if not shooter.has_ammo:
            rewards[agent_idx] += cfg.REWARD_NO_AMMO
            return False, False

        if not shooter.in_range(target):
            rewards[agent_idx] += cfg.REWARD_OUT_OF_RANGE
            return False, False

        fired, hit = shooter.shoot(target, self.rng)
        if fired:
            self.stats["shots_fired"] += 1
        if hit:
            self.stats["hits"] += 1

        return fired, hit

    def _execute_enemy_action(
        self,
        enemy: Vehicle,
        action: int,
        ally_rewards: List[float],
    ) -> None:
        """Ejecuta una acción del enemigo (IA por reglas). Actualiza recompensas aliadas."""
        if action < 5:
            dx, dy = _MOVE_DELTAS[action]
            enemy.move(dx, dy, self.grid_size)
        else:
            ally_idx = action - 5
            if ally_idx < len(self.allies):
                target = self.allies[ally_idx]
                if not target.alive:
                    return
                fired, hit = enemy.shoot(target, self.rng)
                if hit and not target.alive:
                    # El aliado acaba de ser destruido
                    self.stats["ally_losses"] += 1
                    for i in range(cfg.NUM_ALLIES):
                        ally_rewards[i] += cfg.REWARD_ALLY_KILLED

    # ── Observaciones (observabilidad parcial) ────────────────────────────────

    def _get_observations(self) -> List[np.ndarray]:
        """
        Construye el vector de observación para cada agente aliado.

        Estructura (total = OBS_SIZE = 26):
          [0:4]   → propio: x/N, y/N, hp/max_hp, ammo/max_ammo
          [4:14]  → aliados: por cada otro aliado: rel_x, rel_y, hp, ammo, visible (×2)
          [14:26] → enemigos: por cada enemigo: rel_x, rel_y, hp, visible (×3)
        """
        obs_list = []
        for agent_idx, agent in enumerate(self.allies):
            obs = np.zeros(cfg.OBS_SIZE, dtype=np.float32)
            ptr = 0

            # Propio estado (siempre visible)
            if agent.alive:
                obs[ptr:ptr+4] = agent.obs_vector(self.grid_size)
            ptr += 4

            # Aliados
            for other_idx, other in enumerate(self.allies):
                if other_idx == agent_idx:
                    continue
                visible = int(self._is_visible(agent, other))
                if visible and agent.alive:
                    rel_x = (other.x - agent.x) / self.grid_size
                    rel_y = (other.y - agent.y) / self.grid_size
                    obs[ptr:ptr+5] = [
                        rel_x,
                        rel_y,
                        other.health / other.max_health,
                        other.ammo   / other.max_ammo,
                        1.0,
                    ]
                else:
                    obs[ptr:ptr+5] = [0, 0, 0, 0, 0]
                ptr += 5

            # Enemigos
            for enemy in self.enemies:
                visible = int(self._is_visible(agent, enemy) and enemy.alive)
                if visible and agent.alive:
                    rel_x = (enemy.x - agent.x) / self.grid_size
                    rel_y = (enemy.y - agent.y) / self.grid_size
                    obs[ptr:ptr+4] = [
                        rel_x,
                        rel_y,
                        enemy.health / enemy.max_health,
                        1.0,
                    ]
                else:
                    obs[ptr:ptr+4] = [0, 0, 0, 0]
                ptr += 4

            obs_list.append(obs)

        return obs_list

    def _is_visible(self, observer: Vehicle, target: Vehicle) -> bool:
        """Visibilidad basada en distancia Chebyshev (8-direccional)."""
        if not observer.alive:
            return False
        return max(abs(observer.x - target.x), abs(observer.y - target.y)) <= cfg.OBS_RADIUS

    # ── Utilidades ────────────────────────────────────────────────────────────

    @property
    def alive_allies(self) -> List[Vehicle]:
        return [a for a in self.allies if a.alive]

    @property
    def alive_enemies(self) -> List[Vehicle]:
        return [e for e in self.enemies if e.alive]

    def render_ascii(self) -> str:
        """Representación ASCII del estado actual de la cuadrícula."""
        grid = [["·"] * self.grid_size for _ in range(self.grid_size)]

        for ally in self.allies:
            if ally.alive:
                grid[ally.y][ally.x] = f"A{ally.id}"

        for enemy in self.enemies:
            if enemy.alive:
                grid[enemy.y][enemy.x] = f"E{enemy.id}"

        separator = "+" + "---+" * self.grid_size
        lines = [separator]
        for row in grid:
            cells = "|".join(f"{c:^3}" for c in row)
            lines.append(f"|{cells}|")
            lines.append(separator)
        lines.append(f"Paso: {self.step_count}  Aliados vivos: {len(self.alive_allies)}  Enemigos vivos: {len(self.alive_enemies)}")
        return "\n".join(lines)
