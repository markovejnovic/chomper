"""This file exposes utility math functions.

Copyright Marko Vejnovic <contact@markovejnovic.com> 2021
"""

import numpy as np


def normalize_linear(value: np.ndarray) -> np.ndarray:
    """Returns a normalized array."""
    return value / np.max(value)
