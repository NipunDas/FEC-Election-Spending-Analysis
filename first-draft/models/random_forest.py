import constants
import numpy as np
import polars as pl
from .base_model import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from typing import Optional

from typing import Optional

class RandomForestModel(BaseModel):
  """Heuristic model that labels each committee with the party it donated most to, then predicts the most common donations."""

  DEFAULT_INPUT_FEATURES = [
    'election.type',
    'candidate.party',
    'recipient.state',
    'contributor.state',
    'ico.status',
  ]

  DEFAULT_N_ESTIMATORS = 300
  DEFAULT_N_JOBS = -1
  DEFAULT_RANDOM_SEED = 10

  def __init__(self, input_features: Optional[list[str]] = None, n_estimators: Optional[int] = None,
               n_jobs: Optional[int] = None, random_seed: Optional[int] = None):
    super().__init__(name='Random Forest')
    
    self.input_features = input_features if input_features is not None else self.DEFAULT_INPUT_FEATURES
    self.n_estimators = n_estimators if n_estimators is not None else self.DEFAULT_N_ESTIMATORS
    self.n_jobs = n_jobs if n_jobs is not None else self.DEFAULT_N_JOBS
    self.random_seed = random_seed if random_seed is not None else self.DEFAULT_RANDOM_SEED

    cat_input_features = [f for f in self.input_features if f in constants.CATEGORICAL_FEATURES]
    preprocessor = ColumnTransformer(
      transformers = [('categorical_ohe', OneHotEncoder(), cat_input_features)],
      remainder='passthrough',
    )
    rf_regressor = RandomForestRegressor(
      n_estimators=self.n_estimators,
      n_jobs=self.n_jobs,
      random_state=self.random_seed,
    )

    self.model = Pipeline([
      ('preprocessing', preprocessor),
      ('rf_regressor', rf_regressor),
    ])

  def fit(self, X_train: pl.DataFrame, y_train: pl.Series):
    self.model.fit(X_train.select(self.input_features).to_pandas(), y_train.to_pandas())

  def predict(self, X: pl.DataFrame) -> np.ndarray:
    return self.model.predict(X.select(self.input_features).to_pandas())
