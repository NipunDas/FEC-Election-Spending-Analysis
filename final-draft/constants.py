RAW_CONTRIBUTIONS_TABLE_CSV_PATH = './data/contribDB_2010.csv.gz'
RAW_RECIPIENTS_TABLE_CSV_PATH = './data/dime_recipients_1979_2024.csv'
RAW_CONTRIBUTORS_TABLE_CSV_PATH = './data/dime_contributors_1979_2024.csv.gz'
CONTRIBUTIONS_TABLE_PARQUET_PATH = './intermediate-output/contributions.parquet'

TARGET_COL = 'amount'

CANDIDATE_FEATURES = [
  'candidate.id',
  'recipient.name',
  'candidate.party',
  'recipient.state',
  'election.type',
  'ico.status',
]

COMMITTEE_FEATURES = [
  'committee.id',
  'contributor.name',
  'contributor.address',
  'contributor.city',
  'contributor.state',
  'contributor.zipcode',
  'is.corp',
]

TRAIN_FEATURES = CANDIDATE_FEATURES + COMMITTEE_FEATURES

CATEGORICAL_FEATURES = [
  'election.type',
  'candidate.party',
  'recipient.state',
  'contributor.state',
  'ico.status',
]
