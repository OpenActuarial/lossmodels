from abc import ABC, abstractmethod
import numpy as np
from ..utils.random import RNGLike


class FrequencyModel(ABC):
    """
    Base class for all frequency distributions.

    All frequency models must implement:
    - sample
    - mean
    - variance
    """

    @abstractmethod
    def sample(self, size: int = 1, rng: RNGLike = None) -> np.ndarray:
        """
        Generate random samples of claim counts.

        Parameters
        ----------
        size : int
            Number of samples

        Returns
        -------
        np.ndarray
            Array of claim counts
        """
        pass

    @abstractmethod
    def mean(self) -> float:
        """Expected number of claims"""
        pass

    @abstractmethod
    def variance(self) -> float:
        """Variance of number of claims"""
        pass

    def std(self) -> float:
        """Standard deviation of claim count"""
        return np.sqrt(self.variance())

    def __repr__(self):
        return f"{self.__class__.__name__}()"
