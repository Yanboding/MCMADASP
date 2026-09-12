import gurobipy as gp
import numpy as np


class Features:
    __slots__ = ('size', 'blocks')

    def __init__(self, size, blocks=()):
        self.size = int(size)
        self.blocks = list(blocks)

    def scaled(self, factor):
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
        for start, stop, block in self.blocks:
            target[start:stop] += block
        return target

    def dot(self, theta):
        if not isinstance(theta, gp.MVar):
            theta = np.asarray(theta, dtype=float)
        total = 0.0
        for start, stop, block in self.blocks:
            term = theta[start:stop] @ block
            total = total + (term.item() if hasattr(term, 'item') else term)
        return total

    def dense(self):
        return self.add_to(np.zeros(self.size))
