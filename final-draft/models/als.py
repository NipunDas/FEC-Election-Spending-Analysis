import numpy as np
import polars as pl
from .base_model import BaseModel
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
from typing import Optional

class ALSModel(BaseModel):
  """Collaborative Filtering model that analyzes implicit ratings using ALS."""

  DEFAULT_N_FACTORS = 350
  DEFAULT_REGULARIZATION = 0.1
  DEFAULT_N_ITERATIONS = 20
  DEFAULT_RANDOM_SEED = 10

  def __init__(self, n_factors: Optional[int] = None, regularization: Optional[float] = None,
               n_iterations: Optional[int] = None, random_seed: Optional[int] = None):
    super().__init__(name='ALS Collaborative Filtering')

    self.n_factors = n_factors if n_factors is not None else self.DEFAULT_N_FACTORS
    self.regularization = regularization if regularization is not None else self.DEFAULT_REGULARIZATION
    self.n_iterations = n_iterations if n_iterations is not None else self.DEFAULT_N_ITERATIONS
    self.random_seed = random_seed if random_seed is not None else self.DEFAULT_RANDOM_SEED

    self.user_mapping = dict()
    self.item_mapping = dict()
    self.model = AlternatingLeastSquares(
      factors=self.n_factors,
      regularization=self.regularization,
      iterations=self.n_iterations,
      random_state=self.random_seed,
    )

  def fit(self, X_train: pl.DataFrame, y_train: pl.Series):
    agg_donations_df = (X_train
                        .select(['committee.id', 'candidate.id', y_train.alias('amount')])
                        .group_by(['committee.id', 'candidate.id'])
                        .sum())
    
    donor_ids = agg_donations_df['committee.id'].unique().to_list()
    self.user_mapping = {donor_id: idx for idx, donor_id in enumerate(donor_ids)}
    # inv_user_mapping = {idx: donor_id for donor_id, idx in user_mapping.items()}

    candidate_ids = agg_donations_df['candidate.id'].unique().to_list()
    self.item_mapping = {cand_id: idx for idx, cand_id in enumerate(candidate_ids)}
    # inv_item_mapping = {idx: cand_id for cand_id, idx in item_mapping.items()}

    alpha = 40.0
    agg_donations_df = agg_donations_df.with_columns(
      pl.col('committee.id').replace(self.user_mapping).alias('user_idx'),
      pl.col('candidate.id').replace(self.item_mapping).alias('item_idx'),
      # confidence score based on donation amounts
      (1.0 + alpha * (pl.col('amount') + 1.0).log()).alias('confidence')
    )
    
    num_users = len(self.user_mapping)
    num_items = len(self.item_mapping)
    user_item_matrix = coo_matrix(
      (
        agg_donations_df['confidence'].to_numpy().astype('float32'),
        (
          agg_donations_df['user_idx'].to_numpy(),
          agg_donations_df['item_idx'].to_numpy(),
        ),
      ),
      shape=(num_users, num_items),
    ).tocsr()

    self.model.fit(user_item_matrix)

  def predict(self, X: pl.DataFrame) -> np.ndarray:
    predictions = []

    for row in X.iter_rows(named=True):
      committee_id = row['committee.id']
      candidate_id = row['candidate.id']
      
      # cold start case
      if committee_id not in self.user_mapping or candidate_id not in self.item_mapping:
        predictions.append(float(0))
        continue

      user_id = self.user_mapping[row['committee.id']]
      item_id = self.item_mapping[row['candidate.id']]

      #print(len(self.user_mapping))
      #print(len(self.item_mapping))
      #print(len(self.model.user_factors))
      #print(len(self.model.item_factors))
      #exit()
      user_vec = self.model.user_factors[user_id]
      item_vec = self.model.item_factors[item_id]

      predictions.append(float(np.dot(user_vec, item_vec)))

    return np.array(predictions)
