# MultiAgent/__init__.py

from .multi_agent import (
    build_math_agent,
    build_physics_agent,
    build_chemistry_agent,
)

from .leader_reviewer import (
    build_leader_agent,
    build_reviewer_agent,
    classify_subject,
    review_answer,
)

__all__ = [
    "build_math_agent",
    "build_physics_agent",
    "build_chemistry_agent",
    "build_leader_agent",
    "build_reviewer_agent",
    "classify_subject",
    "review_answer",
]