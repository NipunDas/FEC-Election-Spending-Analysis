import numpy as np
import polars as pl
from .base_model import BaseModel
from typing import Optional

class HeuristicModel(BaseModel):
  """Heuristic model that labels each committee with the party it donated most to, then predicts the most common donations."""

  def __init__(self):
    super().__init__(name='Heuristic')
    
    self.committee_party_df: Optional[pl.DataFrame] = None

  def fit(self, X_train: pl.DataFrame, y_train: pl.Series):
    self.committee_party_df = (X_train
                               .with_columns(y_train.alias('amount'))
                               .group_by(['committee.id', 'candidate.party'])
                               .sum()
                               .sort(by='amount', descending=True)
                               .group_by('committee.id', maintain_order=True)
                               .first()[['committee.id', 'candidate.party', 'amount']])

  def predict(self, X: pl.DataFrame) -> np.ndarray:
    if self.committee_party_df is None:
      raise ValueError('Call fit() before attempting to predict!')
    
    X_with_committee_scores = X.join(self.committee_party_df, on=['committee.id', 'candidate.party'], how='left', maintain_order='left')
    return X_with_committee_scores.fill_null(0)['amount'].to_numpy()
