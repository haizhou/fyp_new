"""Compose: a small typed plan algebra opening the composition space.

Grammar-closed, composition-open. Node types are a fixed enum; trees are
arbitrary (depth-capped). Every tree is type-checked before evaluation and
evaluated deterministically over the flat first-party record universe.

This package is an experimental track: it shares the KG loading path with the
frozen v2.2 pipeline but touches none of its planning/verification code.
"""

from .algebra import validate_tree, AlgebraError
from .eval_runtime import RuntimeAlgebraEvaluator

__all__ = ["validate_tree", "AlgebraError", "RuntimeAlgebraEvaluator"]
