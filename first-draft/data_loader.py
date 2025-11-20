import constants
import polars as pl
from typing import Optional

class DataLoader:
  """A utility class that loads data for model evaluation."""

  DEFAULT_TEST_SPLIT_PERCENT = 0.2

  def __init__(self, test_split_percent: Optional[float] = None):
    self.test_split_percent = test_split_percent or self.DEFAULT_TEST_SPLIT_PERCENT

    self.train_df: Optional[pl.DataFrame] = None
    self.committee_df: Optional[pl.DataFrame] = None
    self.candidate_df: Optional[pl.DataFrame] = None
    self.eval_candidates: Optional[list[str]] = None
    self.relevant_committees_per_candidate: dict[str, list[str]] = dict()

  def prepare_data(self):
    sorted_contributions_df = pl.read_parquet(constants.CONTRIBUTIONS_TABLE_PARQUET_PATH).sort(by='date', descending=True)
    n_test_rows = round(self.test_split_percent * sorted_contributions_df.height)
    held_out_contributions_df = sorted_contributions_df.head(n_test_rows)

    self.train_df = sorted_contributions_df.slice(n_test_rows)
    self.committee_df = sorted_contributions_df.group_by('committee.id').first().select(constants.COMMITTEE_FEATURES)
    self.candidate_df = sorted_contributions_df.group_by('candidate.id').first().select(constants.CANDIDATE_FEATURES)
    self.eval_candidates = held_out_contributions_df['candidate.id'].unique().sort().to_list()

    relevant_committees_per_candidate_df = (held_out_contributions_df
                                            .group_by('candidate.id')
                                            .agg(
                                              pl.col('committee.id').unique().sort().implode().alias('relevant_committee_ids')
                                            ))
    for row in relevant_committees_per_candidate_df.iter_rows(named = True):
      self.relevant_committees_per_candidate[row['candidate.id']] = row['relevant_committee_ids']

  def get_training_data(self) -> tuple[pl.DataFrame, pl.Series]:
    if self.train_df is None:
      raise ValueError('Must call prepare_data() first!')

    return (self.train_df.select(constants.TRAIN_FEATURES), self.train_df[constants.TARGET_COL])
  
  def get_eval_candidates(self) -> list[str]:
    if self.eval_candidates is None:
      raise ValueError('Must call prepare_data() first!')
    
    return self.eval_candidates
  
  def get_potential_contributions_df(self, candidate_id: str) -> pl.DataFrame:
    if self.committee_df is None or self.candidate_df is None:
      raise ValueError('Must call prepare_data() first!')
    
    candidate_features_row = self.candidate_df.filter(pl.col('candidate.id') == candidate_id)

    if candidate_features_row.height != 1:
      raise ValueError(f'Invalid candidate ID: {candidate_id}')
    
    return candidate_features_row.join(self.committee_df, how='cross')
