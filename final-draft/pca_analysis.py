"""
A script that analyzes candidate embeddings using PCA.
"""
import constants
import matplotlib.pyplot as plt
import polars as pl
from sklearn.decomposition import PCA

def main():
  print('Analyzing candidate embeddings using PCA...')
  cand_embeddings_df = pl.read_parquet(constants.CANDIDATE_EMBEDDINGS_PARQUET_PATH)
  X = cand_embeddings_df.drop('cand_id').to_pandas()

  pca = PCA()
  pca.fit(X)

  # Plot scree plot using PCA analysis
  explained_variance_ratio = pca.explained_variance_ratio_

  plt.figure(figsize=(10, 6))
  plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
  plt.title('Scree Plot of Explained Variance')
  plt.xlabel('Principal Component Number')
  plt.ylabel('Proportion of Variance Explained')
  plt.grid(True)
  plt.show()


if __name__ == '__main__':
  main()
