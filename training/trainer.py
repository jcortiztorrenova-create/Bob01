"""
Bucle de entrenamiento para MARL táctico con IQL (Independent Q-Learning).

Flujo por episodio:
  reset() → loop: obs → acciones → step() → store() → train_step()
  → decay_epsilon() → [evaluate cada EVAL_INTERVAL episodios]
  → [guardar checkpoints cada SAVE_INTERVAL episodios]
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import List, Optional

import numpy as np

import config as cfg
from agents.dqn_agent import DQNAgent
from agents.enemy_ai import RuleBasedEnemyAI
from marl_tactical.environment import TacticalEnv


class MARLTrainer:
    """Orquesta el entrenamiento de N agentes DQN en el entorno táctico POSG."""

    def __init__(self, seed: Optional[int] = None) -> None:
        seed = seed or cfg.SEED
        np.random.seed(seed)

        self.enemy_ai = RuleBasedEnemyAI()
        self.env      = TacticalEnv(enemy_ai=self.enemy_ai, seed=seed)

        self.agents: List[DQNAgent] = [
            DQNAgent(agent_id=i) for i in range(cfg.NUM_ALLIES)
        ]

        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cfg.LOG_DIR, exist_ok=True)

        # Métricas de entrenamiento
        self._win_history:    deque = deque(maxlen=100)
        self._reward_history: deque = deque(maxlen=100)
        self._loss_history:   deque = deque(maxlen=100)

    # ── Entrenamiento principal ───────────────────────────────────────────────

    def train(self) -> None:
        """Ejecuta el bucle de entrenamiento completo."""
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"  MARL Táctico — Entrenamiento IQL/DQN")
        print(f"  Episodios: {cfg.MAX_EPISODES}  |  Agentes: {cfg.NUM_ALLIES}  |  Enemigos: {cfg.NUM_ENEMIES}")
        print(f"  Cuadrícula: {cfg.GRID_SIZE}×{cfg.GRID_SIZE}  |  Dispositivo: {self.agents[0].device}")
        print(f"{'='*60}\n")

        for episode in range(1, cfg.MAX_EPISODES + 1):
            ep_rewards, ep_win = self._run_episode(training=True)
            self._win_history.append(float(ep_win))
            self._reward_history.append(float(np.mean(ep_rewards)))

            # Decaimiento de epsilon al final de cada episodio
            for agent in self.agents:
                agent.decay_epsilon()

            # Log periódico
            if episode % 100 == 0:
                elapsed   = time.time() - start_time
                win_rate  = np.mean(self._win_history) * 100
                mean_rew  = np.mean(self._reward_history)
                epsilon   = self.agents[0].epsilon
                buff_size = len(self.agents[0].buffer)
                print(
                    f"Ep {episode:5d}/{cfg.MAX_EPISODES}  "
                    f"WinRate: {win_rate:5.1f}%  "
                    f"Reward: {mean_rew:+7.2f}  "
                    f"ε: {epsilon:.3f}  "
                    f"Buffer: {buff_size:6d}  "
                    f"Tiempo: {elapsed:6.0f}s"
                )

            # Evaluación periódica
            if episode % cfg.EVAL_INTERVAL == 0:
                self._evaluate(episode)

            # Guardado de checkpoints
            if episode % cfg.SAVE_INTERVAL == 0:
                self._save_checkpoints(episode)

        print("\nEntrenamiento completado.")
        self._save_checkpoints(cfg.MAX_EPISODES)

    # ── Episodio único ────────────────────────────────────────────────────────

    def _run_episode(self, training: bool = True) -> tuple:
        """
        Ejecuta un episodio completo.

        Args:
            training: si True, explora (ε-greedy) y entrena las redes.

        Returns:
            (cumulative_rewards_per_agent, win)
        """
        observations = self.env.reset()
        cumulative   = np.zeros(cfg.NUM_ALLIES)
        done         = False

        while not done:
            # Selección de acciones (ε-greedy en entrenamiento, greedy en eval)
            actions = [
                agent.select_action(obs, greedy=not training)
                for agent, obs in zip(self.agents, observations)
            ]

            next_obs, rewards, done, info = self.env.step(actions)

            if training:
                # Almacenar transiciones
                for i, agent in enumerate(self.agents):
                    agent.store(
                        observations[i], actions[i],
                        rewards[i],
                        next_obs[i], done,
                    )
                    # Paso de gradiente
                    loss = agent.train_step()
                    if loss is not None:
                        self._loss_history.append(loss)

            cumulative   += np.array(rewards)
            observations  = next_obs

        win = len(self.env.alive_enemies) == 0
        return cumulative, win

    # ── Evaluación ────────────────────────────────────────────────────────────

    def _evaluate(self, episode: int) -> None:
        """Evaluación sin exploración sobre EVAL_EPISODES episodios."""
        wins    = 0
        rewards = []

        for _ in range(cfg.EVAL_EPISODES):
            ep_rewards, win = self._run_episode(training=False)
            wins    += int(win)
            rewards.append(np.mean(ep_rewards))

        win_rate  = wins / cfg.EVAL_EPISODES * 100
        mean_rew  = np.mean(rewards)

        print(f"\n  [EVAL ep {episode}]  "
              f"WinRate: {win_rate:.1f}%  "
              f"Reward medio: {mean_rew:+.2f}\n")

        # Guardar log
        log_path = os.path.join(cfg.LOG_DIR, "eval_log.csv")
        header   = not os.path.exists(log_path)
        with open(log_path, "a") as f:
            if header:
                f.write("episode,win_rate,mean_reward\n")
            f.write(f"{episode},{win_rate:.2f},{mean_rew:.4f}\n")

    # ── Persistencia ─────────────────────────────────────────────────────────

    def _save_checkpoints(self, episode: int) -> None:
        for agent in self.agents:
            path = os.path.join(
                cfg.CHECKPOINT_DIR, f"agent_{agent.agent_id}_ep{episode}.pt"
            )
            agent.save(path)
        print(f"  [Checkpoint guardado — episodio {episode}]")

    def load_checkpoints(self, episode: int) -> None:
        for agent in self.agents:
            path = os.path.join(
                cfg.CHECKPOINT_DIR, f"agent_{agent.agent_id}_ep{episode}.pt"
            )
            if os.path.exists(path):
                agent.load(path)
                print(f"  Cargado: {path}")
            else:
                print(f"  [AVISO] No encontrado: {path}")
