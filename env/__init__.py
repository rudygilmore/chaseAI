"""Chase environment package."""

from .chase_env import CANVAS_SIZE, ChaseEnv, register_chase_v0

register_chase_v0()

__all__ = ["CANVAS_SIZE", "ChaseEnv", "register_chase_v0"]
