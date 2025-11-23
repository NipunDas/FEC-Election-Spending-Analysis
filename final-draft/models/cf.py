import numpy as np
import polars as pl
from .base_model import BaseModel
from surprise import SVD, Dataset, Reader
from typing import Optional

class CFModel(BaseModel):
  """Collaborative Filtering model that uses surprise with SVD matrix factorization."""

  DEFAULT_N_FACTORS = 50
  DEFAULT_N_EPOCHS = 50
  DEFAULT_LR_ALL = 0.01
  DEFAULT_REG_ALL = 0.005
  DEFAULT_RANDOM_SEED = 10

  def __init__(self, n_factors: Optional[int] = None, n_epochs: Optional[int] = None,
               lr_all: Optional[float] = None, reg_all: Optional[float] = None, random_seed: Optional[int] = None):
    super().__init__(name='Collaborative Filtering')

    self.n_factors = n_factors if n_factors is not None else self.DEFAULT_N_FACTORS
    self.n_epochs = n_epochs if n_epochs is not None else self.DEFAULT_N_EPOCHS
    self.lr_all = lr_all if lr_all is not None else self.DEFAULT_LR_ALL
    self.reg_all = reg_all if reg_all is not None else self.DEFAULT_REG_ALL
    self.random_seed = random_seed if random_seed is not None else self.DEFAULT_RANDOM_SEED

    self.model = SVD(
      n_factors=self.n_factors,
      n_epochs=self.n_epochs,
      lr_all=self.lr_all,
      reg_all=self.reg_all,
      random_state=self.random_seed,
    )

  def fit(self, X_train: pl.DataFrame, y_train: pl.Series):
    agg_donations_df = (X_train
                        .select(['committee.id', 'candidate.id', y_train.alias('amount')])
                        .group_by(['committee.id', 'candidate.id'])
                        .sum())
    # min-max scale by committee (each committee's minimum donation is 0, max donation is 1)
    normalized_donations_df = (agg_donations_df
                               .with_columns(
                                  pl.col('amount').min().over('committee.id').alias('min_donation'),
                                  pl.col('amount').max().over('committee.id').alias('max_donation'),
                                )
                                .with_columns(
                                  ((pl.col('amount') - pl.col('min_donation')) / (pl.col('max_donation') - pl.col('min_donation')))
                                  .clip(0, 1).alias('normalized_donation')
                                )
                                .select(
                                  pl.col('committee.id').alias('userID'),
                                  pl.col('candidate.id').alias('itemID'),
                                  pl.col('normalized_donation').alias('rating'),
                                ))
    
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(normalized_donations_df.to_pandas(), reader)
    trainset = data.build_full_trainset()

    self.model.fit(trainset)

  def predict(self, X: pl.DataFrame) -> np.ndarray:
    predictions = []

    for row in X.iter_rows(named=True):
      user_id = row['committee.id']
      item_id = row['candidate.id']
      pred = self.model.predict(user_id, item_id, verbose=False)
      predictions.append(np.clip(pred.est, 0.0, 1.0))

    return np.array(predictions)
