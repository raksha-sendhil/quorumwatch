"""LuxR-inspired cooperative activation."""
import numpy as np


def hill(A, K, n):
    """H(A) = A^n / (K^n + A^n), in [0, 1)."""
    A = np.maximum(A, 0.0)
    An = A ** n
    return An / (K ** n + An)


def hill_max_slope(K, n):
    """Largest value of dH/dA, attained at the inflection point."""
    if n <= 1:
        return n / K
    x = ((n - 1) / (n + 1)) ** (1 / n)
    return n * x ** (n - 1) / (K * (1 + x ** n) ** 2)
