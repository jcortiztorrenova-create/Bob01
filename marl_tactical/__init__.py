"""Paquete principal del entorno MARL Táctico (POSG)."""
from .entities import Vehicle, Team
from .environment import TacticalEnv

__all__ = ["Vehicle", "Team", "TacticalEnv"]
