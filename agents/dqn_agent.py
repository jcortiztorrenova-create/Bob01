"""
Agente DQN (Deep Q-Network) independiente para MARL táctico.

Arquitectura:
  - Red principal (online network): estima Q(s, a)
  - Red objetivo  (target network): proporciona Q-targets estables
  - Actualización suave (soft update): θ_target ← τ·θ + (1-τ)·θ_target

Política de exploración:
  - ε-greedy con decaimiento multiplicativo por episodio
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import config as cfg
from agents.replay_buffer import ReplayBuffer


# ── Red neuronal ──────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Red completamente conectada que aproxima Q(s, a).
    Entrada:  vector de observación de tamaño OBS_SIZE
    Salida:   NUM_ACTIONS valores Q
    """

    def __init__(self, obs_size: int, num_actions: int) -> None:
        super().__init__()
        layers = []
        in_dim = obs_size
        for out_dim in cfg.HIDDEN_LAYERS:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Agente DQN ────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Agente DQN independiente (IQL — Independent Q-Learning).

    Cada aliado tiene su propia red y buffer; aprenden en paralelo sin
    comunicación explícita (extensible a CTDE en iteraciones futuras).
    """

    def __init__(
        self,
        agent_id:    int,
        obs_size:    int = cfg.OBS_SIZE,
        num_actions: int = cfg.NUM_ACTIONS,
        device:      Optional[str] = None,
    ) -> None:
        self.agent_id    = agent_id
        self.num_actions = num_actions
        self.device      = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.online_net = QNetwork(obs_size, num_actions).to(self.device)
        self.target_net = QNetwork(obs_size, num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=cfg.LEARNING_RATE)
        self.buffer    = ReplayBuffer(cfg.REPLAY_CAPACITY)

        self.epsilon      = cfg.EPS_START
        self.total_steps  = 0
        self.train_steps  = 0

    # ── Selección de acción ───────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """
        Selecciona acción usando ε-greedy.
        Si greedy=True, usa política greedy pura (evaluación).
        """
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state)
        return int(q_values.argmax(dim=1).item())

    # ── Almacenamiento de experiencia ─────────────────────────────────────────

    def store(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        self.buffer.push(obs, action, reward, next_obs, done)
        self.total_steps += 1

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def train_step(self) -> Optional[float]:
        """
        Ejecuta un paso de gradiente si el buffer tiene suficientes muestras.
        Retorna la pérdida escalar o None si no hay suficientes datos.
        """
        if not self.buffer.ready:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(cfg.BATCH_SIZE)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q(s,a) actuales
        q_current = self.online_net(states).gather(1, actions)

        # Q-targets: r + γ · max_a' Q_target(s', a')  (0 si done)
        with torch.no_grad():
            q_next   = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            q_target = rewards + cfg.GAMMA * q_next * (1 - dones)

        loss = F.huber_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_steps += 1

        # Actualización suave de la red objetivo
        if self.train_steps % cfg.TARGET_UPDATE == 0:
            self._soft_update()

        return float(loss.item())

    def _soft_update(self) -> None:
        """θ_target ← τ·θ_online + (1-τ)·θ_target"""
        for p_online, p_target in zip(
            self.online_net.parameters(), self.target_net.parameters()
        ):
            p_target.data.copy_(cfg.TAU * p_online.data + (1 - cfg.TAU) * p_target.data)

    def decay_epsilon(self) -> None:
        """Reduce ε al final de cada episodio."""
        self.epsilon = max(cfg.EPS_END, self.epsilon * cfg.EPS_DECAY)

    # ── Persistencia ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "online":   self.online_net.state_dict(),
            "target":   self.target_net.state_dict(),
            "optim":    self.optimizer.state_dict(),
            "epsilon":  self.epsilon,
            "steps":    self.total_steps,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optim"])
        self.epsilon     = ckpt.get("epsilon", cfg.EPS_END)
        self.total_steps = ckpt.get("steps",   0)
