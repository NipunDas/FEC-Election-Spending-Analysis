import numpy as np
import polars as pl
from abc import ABC, abstractmethod

class BaseModel(ABC):
  """Base class for all models."""

  def __init__(self, name: str):
    self.name = name
    self.model = None

  @abstractmethod
  def fit(self, X_train: pl.DataFrame, y_train: pl.Series):
    """Train the model."""
    pass

  @abstractmethod
  def predict(self, X: pl.DataFrame) -> np.ndarray:
    """Predict donation amounts for each candidate/committee combination."""
    pass

  def recommend(self, X_potential_donations: pl.DataFrame) -> list[str]:
    """Generate ranked list for predicted donations, based on donation amount."""
    potential_donating_committees = X_potential_donations['committee.id'].to_list()
    scores = self.predict(X_potential_donations)
    order = np.argsort(-scores)
    return [potential_donating_committees[i] for i in order]
  