"""
Visualización del entorno MARL táctico.

Funciones:
  - render_episode(): ejecuta y muestra un episodio en ASCII paso a paso.
  - plot_training_curves(): gráficas de tasa de victoria y recompensa media
                            a partir del log de evaluación.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np


# ── Renderizado ASCII en tiempo real ─────────────────────────────────────────

def render_episode(
    env,
    agents: List,
    delay: float = 0.3,
    greedy: bool = True,
) -> dict:
    """
    Ejecuta un episodio completo con renderizado ASCII en la terminal.

    Args:
        env:    instancia de TacticalEnv (ya inicializada con enemy_ai).
        agents: lista de DQNAgent aliados.
        delay:  segundos de pausa entre pasos (0 para deshabilitar).
        greedy: si True, usa política greedy (sin exploración).

    Returns:
        dict con estadísticas del episodio.
    """
    import config as cfg

    obs  = env.reset()
    done = False

    print("\n" + "="*50)
    print("  EPISODIO DE DEMOSTRACIÓN")
    print("="*50)
    print(env.render_ascii())

    while not done:
        actions = [
            agent.select_action(o, greedy=greedy)
            for agent, o in zip(agents, obs)
        ]

        obs, rewards, done, info = env.step(actions)

        # Descripción de acciones
        action_names = ["↑", "↓", "←", "→", "·", "🔫E0", "🔫E1", "🔫E2"]
        action_str = "  ".join(
            f"A{i}:{action_names[a]}" for i, a in enumerate(actions)
        )
        print(f"\nAcciones: {action_str}  |  Recompensas: {[f'{r:+.2f}' for r in rewards]}")
        print(env.render_ascii())

        if delay > 0:
            time.sleep(delay)

    result = "VICTORIA" if len(env.alive_enemies) == 0 else (
        "DERROTA"   if len(env.alive_allies) == 0  else "TIEMPO AGOTADO"
    )
    print(f"\n{'='*50}")
    print(f"  Resultado: {result}")
    print(f"  Pasos: {info['step']}  |  Bajas enemigas: {info['enemy_kills']}  |  Bajas aliadas: {info['ally_losses']}")
    print(f"  Disparos realizados: {info['shots_fired']}  |  Impactos: {info['hits']}")
    if info['shots_fired'] > 0:
        print(f"  Precisión: {info['hits']/info['shots_fired']*100:.1f}%")
    print("="*50)

    return info


# ── Curvas de entrenamiento ────────────────────────────────────────────────────

def plot_training_curves(log_path: Optional[str] = None) -> None:
    """
    Genera gráficas de tasa de victoria y recompensa media desde el CSV de evaluación.

    Requiere matplotlib. Si no está disponible, imprime los valores en texto.

    Args:
        log_path: ruta al CSV generado por MARLTrainer. Por defecto 'logs/eval_log.csv'.
    """
    import config as cfg
    log_path = log_path or os.path.join(cfg.LOG_DIR, "eval_log.csv")

    if not os.path.exists(log_path):
        print(f"[Visualizador] No se encontró el log: {log_path}")
        return

    episodes, win_rates, mean_rewards = [], [], []
    with open(log_path) as f:
        next(f)  # saltar cabecera
        for line in f:
            ep, wr, mr = line.strip().split(",")
            episodes.append(int(ep))
            win_rates.append(float(wr))
            mean_rewards.append(float(mr))

    try:
        import matplotlib
        matplotlib.use("Agg")   # sin interfaz gráfica (compatible con servidores)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle("MARL Táctico — Curvas de Entrenamiento", fontsize=14)

        axes[0].plot(episodes, win_rates, color="steelblue", linewidth=2)
        axes[0].axhline(50, color="gray", linestyle="--", linewidth=1, label="50%")
        axes[0].set_ylabel("Tasa de victoria (%)")
        axes[0].set_title("Tasa de victoria (evaluación)")
        axes[0].set_ylim(0, 105)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(episodes, mean_rewards, color="coral", linewidth=2)
        axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Episodio")
        axes[1].set_ylabel("Recompensa media")
        axes[1].set_title("Recompensa media por agente (evaluación)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(cfg.LOG_DIR, "training_curves.png")
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"[Visualizador] Gráfica guardada en: {out_path}")

    except ImportError:
        print("[Visualizador] matplotlib no disponible. Resumen en texto:\n")
        print(f"{'Episodio':>10}  {'WinRate (%)':>12}  {'Reward medio':>14}")
        print("-" * 42)
        for ep, wr, mr in zip(episodes, win_rates, mean_rewards):
            print(f"{ep:>10}  {wr:>12.1f}  {mr:>14.4f}")
