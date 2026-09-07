"""Block-sparse feature vectors.

A penalty feature vector ``phi`` in ``R^n`` is stored as a list of
``(start, stop, block)`` slices, where ``block`` is a numpy array (numeric
evaluation) or a Gurobi ``MVar`` / ``MLinExpr`` (symbolic models). Keeping the
blocks lets a model builder accumulate a path's features with one slice-add
per block -- ``target[start:stop] += block`` -- instead of materialising a
dense object array of scalar expressions, which is what keeps the LP
subproblems as cheap to build as before the refactor.
"""
import gurobipy as gp
import numpy as np


class Features:
    __slots__ = ('size', 'blocks')

    def __init__(self, size, blocks=()):
        self.size = int(size)
        self.blocks = list(blocks)

    def scaled(self, factor):
        """``factor * phi``; a zero factor drops every block."""
        if factor == 0.0:
            return Features(self.size)
        return Features(self.size, [(start, stop, factor * block) for start, stop, block in self.blocks])

    def __add__(self, other):
        if other.size != self.size:
            raise ValueError(f"feature sizes differ: {self.size} vs {other.size}")
        return Features(self.size, self.blocks + other.blocks)

    def __sub__(self, other):
        return self + other.scaled(-1.0)

    def add_to(self, target):
        """Accumulate into ``target`` (a numpy vector or ``MLinExpr.zeros(n)``)."""
        for start, stop, block in self.blocks:
            target[start:stop] += block
        return target

    def dot(self, theta):
        """``theta . phi`` for numeric or ``MVar`` coefficients ``theta``."""
        if not isinstance(theta, gp.MVar):
            theta = np.asarray(theta, dtype=float)
        total = 0.0
        for start, stop, block in self.blocks:
            term = theta[start:stop] @ block
            total = total + (term.item() if hasattr(term, 'item') else term)
        return total

    def dense(self):
        """Numeric blocks only: the plain ``numpy`` vector."""
        return self.add_to(np.zeros(self.size))
