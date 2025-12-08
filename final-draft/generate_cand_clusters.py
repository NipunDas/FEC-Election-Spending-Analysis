"""
A script that clusters users using K-means clustering.
"""
import constants
from sklearn.cluster import KMeans
import polars as pl

K = 3

def main():
  print('Reading candidate embedding dataframe...')
  cand_embedding_df = pl.read_parquet(constants.CANDIDATE_EMBEDDINGS_PARQUET_PATH)
  X = cand_embedding_df.drop('cand_id').to_pandas()

  print(f'Clustering candidates with k={K}...')
  k_means = KMeans(n_clusters=K, random_state=10)
  labels = k_means.fit_predict(X)
  cand_embedding_df_clustered = cand_embedding_df.with_columns(cluster = labels)
  
  print('Writing dataframe with cluster labels to parquet...')
  cand_embedding_df_clustered.write_parquet(constants.CANDIDATE_EMBEDDINGS_WITH_CLUSTERS_PARQUET_PATH)


if __name__ == '__main__':
  main()
