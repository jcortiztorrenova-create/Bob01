"""
Script de evaluación y demostración del modelo MARL táctico.

Uso:
    python evaluate.py                          # Carga el último checkpoint y ejecuta demo
    python evaluate.py --episode 5000           # Carga checkpoint del episodio 5000
    python evaluate.py --random                 # Agentes con política aleatoria (baseline)
    python evaluate.py --plot                   # Genera gráficas de entrenamiento
    python evaluate.py --n 50                   # Evaluación cuantitativa sobre 50 episodios
"""

import argparse
import os
import numpy as np

import config as cfg
from agents.dqn_agent import DQNAgent
from agents.enemy_ai import RuleBasedEnemyAI
from marl_tactical.environment import TacticalEnv
from utils.visualizer import render_episode, plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluación del modelo MARL táctico")
    parser.add_argument("--episode", type=int, default=None,
                        help="Episodio del checkpoint a cargar")
    parser.add_argument("--random", action="store_true",
                        help="Usar política aleatoria (baseline)")
    parser.add_argument("--plot", action="store_true",
                        help="Generar curvas de entrenamiento")
    parser.add_argument("--n", type=int, default=1,
                        help="Número de episodios de evaluación")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Pausa entre pasos en la demo ASCII (segundos)")
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.plot:
        plot_training_curves()
        return

    # Construir agentes
    agents = [DQNAgent(agent_id=i) for i in range(cfg.NUM_ALLIES)]

    if not args.random:
        # Determinar qué checkpoint cargar
        episode = args.episode
        if episode is None:
            # Buscar el checkpoint más reciente
            checkpoints = [
                f for f in os.listdir(cfg.CHECKPOINT_DIR)
                if f.startswith("agent_0_ep") and f.endswith(".pt")
            ] if os.path.isdir(cfg.CHECKPOINT_DIR) else []
            if checkpoints:
                episode = max(
                    int(f.replace("agent_0_ep", "").replace(".pt", ""))
                    for f in checkpoints
                )
                print(f"Cargando checkpoint del episodio {episode}...")
            else:
                print("[AVISO] No se encontraron checkpoints. Usando política aleatoria.")
                args.random = True

        if not args.random:
            for agent in agents:
                path = os.path.join(cfg.CHECKPOINT_DIR, f"agent_{agent.agent_id}_ep{episode}.pt")
                if os.path.exists(path):
                    agent.load(path)
                    agent.epsilon = 0.0   # sin exploración en evaluación
                else:
                    print(f"[AVISO] No encontrado: {path}. Usando política aleatoria para A{agent.agent_id}.")
    else:
        for agent in agents:
            agent.epsilon = 1.0   # totalmente aleatorio

    # Entorno
    enemy_ai = RuleBasedEnemyAI()
    env      = TacticalEnv(enemy_ai=enemy_ai, seed=args.seed)

    if args.n == 1:
        # Demo visual
        render_episode(env, agents, delay=args.delay, greedy=not args.random)
    else:
        # Evaluación cuantitativa
        wins, losses, timeouts = 0, 0, 0
        total_rewards = np.zeros(cfg.NUM_ALLIES)

        for ep in range(args.n):
            obs  = env.reset()
            done = False
            ep_rewards = np.zeros(cfg.NUM_ALLIES)

            while not done:
                actions = [
                    agent.select_action(o, greedy=not args.random)
                    for agent, o in zip(agents, obs)
                ]
                obs, rewards, done, info = env.step(actions)
                ep_rewards += np.array(rewards)

            total_rewards += ep_rewards
            if   len(env.alive_enemies) == 0: wins    += 1
            elif len(env.alive_allies)  == 0: losses  += 1
            else:                              timeouts += 1

        n = args.n
        print(f"\n{'='*50}")
        print(f"  Evaluación cuantitativa ({n} episodios)")
        print(f"{'='*50}")
        print(f"  Victorias:        {wins:3d} ({wins/n*100:5.1f}%)")
        print(f"  Derrotas:         {losses:3d} ({losses/n*100:5.1f}%)")
        print(f"  Tiempo agotado:   {timeouts:3d} ({timeouts/n*100:5.1f}%)")
        print(f"  Recompensa media: {np.mean(total_rewards/n):+.2f}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
