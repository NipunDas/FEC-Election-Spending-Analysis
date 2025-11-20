import constants
import polars as pl
from typing import Optional

class DataLoader:
  """A utility class that loads data for model evaluation."""

  DEFAULT_TEST_SPLIT_PERCENT = 0.2

  def __init__(self, test_split_percent: Optional[float] = None):
    self.test_split_percent = test_split_percent or self.DEFAULT_TEST_SPLIT_PERCENT

  def prepare_data(self):
    sorted_contributions_df = pl.read_parquet(constants.CONTRIBUTIONS_TABLE_PARQUET_PATH).sort(by='date', descending=True)
    n_test_rows = round(self.test_split_percent * sorted_contributions_df.height)
    held_out_contributions_df = sorted_contributions_df.head(n_test_rows)
    self.train_df = sorted_contributions_df.slice(n_test_rows)   
