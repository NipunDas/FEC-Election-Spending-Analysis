"""
Script that converts raw contributions table to parquet file containing engineered features for committee contributions.
"""
import constants
import polars as pl

PARTY_CODE_MAP = {
  100: 'Democrat',
  200: 'Republican',
  328: 'Independent',
}

def main():
  contributions_df = pl.read_csv(constants.RAW_CONTRIBUTIONS_TABLE_CSV_PATH, ignore_errors=True)
  recipients_df = pl.read_csv(constants.RAW_RECIPIENTS_TABLE_CSV_PATH, ignore_errors=True)

  # Specifically look at contributions from committees to political candidates for federal congress races
  senate_contributions_df = (contributions_df
                            .filter((pl.col('recipient.type') == 'CAND') & (pl.col('contributor.type') == 'C'))
                            .filter(pl.col('seat').is_in(['federal:house', 'federal:senate']))
                            .filter(pl.col('amount') > 0)
                            .with_columns(date = pl.col('date').str.strptime(pl.Date, '%Y-%m-%d'))
                            .join(recipients_df, on=['bonica.rid', 'cycle'], how='left')
                            .select([
                              # Contribution features
                              'transaction.id',
                              'transaction.type',
                              'date',
                              'amount',
                              # Committee (donor) features
                              pl.col('bonica.cid').alias('committee.id'),
                              'contributor.name',
                              'contributor.address',
                              'contributor.city',
                              'contributor.state',
                              'contributor.zipcode',
                              'is.corp',
                              # Candidate features
                              pl.col('bonica.rid').alias('candidate.id'),
                              'recipient.name',
                              pl.col('party').map_elements(lambda x: PARTY_CODE_MAP.get(x, 'Other')).fill_null('Other').alias('candidate.party'),
                              'recipient.state',
                              'election.type',
                              'ico.status',
                            ]))
  
  senate_contributions_df.write_parquet(constants.CONTRIBUTIONS_TABLE_PARQUET_PATH)


if __name__ == '__main__':
  main()
