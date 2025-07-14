from __future__ import annotations

import abc
from typing import Protocol

import pandas as pd


class Strategy(abc.ABC):
    """Abstract base class – every strategy yields binary positions (+1 / 0 / -1)."""

    name: str = "AbstractStrategy"

    @abc.abstractmethod
    def generate_signals(self, price: pd.Series) -> pd.Series:
        """Return a Series (index=dates) of positions."""
        raise NotImplementedError
