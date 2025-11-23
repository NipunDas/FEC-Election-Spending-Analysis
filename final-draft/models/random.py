import numpy as np
import polars as pl
from .base_model import BaseModel
from typing import Optional

class RandomModel(BaseModel):
  """Random baseline model."""

  DEFAULT_RANDOM_SEED = 10

  def __init__(self, random_seed: Optional[int] = None):
    super().__init__(name='Random')
    self.random_seed = random_seed if random_seed is not None else self.DEFAULT_RANDOM_SEED
    self.rng = np.random.RandomState(self.random_seed)

  def fit(self, X_train: pl.DataFrame, y_train: pl.Series):
    pass

  def predict(self, X: pl.DataFrame) -> np.ndarray:
    return self.rng.rand(len(X))
