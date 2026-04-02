"""
Punto de entrada principal — Entrenamiento MARL Táctico (POSG + DQN).

Uso:
    python main.py                  # Entrenamiento completo
    python main.py --episodes 1000  # Entrenamiento corto (prueba)
    python main.py --seed 7         # Semilla aleatoria específica
"""

import argparse
import config as cfg
from training.trainer import MARLTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento MARL táctico basado en POSG + DQN"
    )
    parser.add_argument(
        "--episodes", type=int, default=cfg.MAX_EPISODES,
        help=f"Número de episodios de entrenamiento (default: {cfg.MAX_EPISODES})"
    )
    parser.add_argument(
        "--seed", type=int, default=cfg.SEED,
        help=f"Semilla aleatoria (default: {cfg.SEED})"
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Desactivar evaluaciones periódicas"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ajuste dinámico de parámetros de entrenamiento
    cfg.MAX_EPISODES = args.episodes
    cfg.SEED         = args.seed
    if args.no_eval:
        cfg.EVAL_INTERVAL = args.episodes + 1   # nunca evalúa

    trainer = MARLTrainer(seed=args.seed)
    trainer.train()


if __name__ == "__main__":
    main()
