# MultiAgent/__init__.py

from .multi_agent import (
    build_math_agent,
    build_physics_agent,
    build_chemistry_agent,
)

__all__ = ["build_math_agent", "build_physics_agent", "build_chemistry_agent"]