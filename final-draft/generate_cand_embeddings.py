"""
A script that generates candidate embeddings from ALS collaborative filtering.
"""
import constants
import polars as pl
from data_loader import DataLoader
from models.als import ALSModel

def main():
  print('Loading and preparing data...')
  data_loader = DataLoader()
  data_loader.prepare_data()

  als_model = ALSModel(n_factors=4) # n_factors determined using PCA

  X_train, y_train = data_loader.get_training_data()

  print('Fitting ALS model...')
  als_model.fit(X_train, y_train)

  print('Getting candidate embeddings...')
  embedding_rows = []
  
  for cand_id, idx in als_model.item_mapping.items():
    row = { 'cand_id': cand_id }
    factors = als_model.model.item_factors[idx]

    for n, factor in enumerate(factors):
      row[f'factor_{n}'] = factor
    embedding_rows.append(row)
  
  embedding_df = pl.DataFrame(embedding_rows)
  
  print('Writing candidate embeddings to parquet...')
  embedding_df.write_parquet(constants.CANDIDATE_EMBEDDINGS_PARQUET_PATH)
    

if __name__ == '__main__':
  main()
