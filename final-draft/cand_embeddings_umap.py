"""
A script that visualizes candidate embeddings using UMAP.
"""
import constants
import matplotlib.pyplot as plt
import polars as pl
import umap

def main():
  print('Visualizing candidate embeddings using UMAP...')
  cand_embeddings_df = pl.read_parquet(constants.CANDIDATE_EMBEDDINGS_WITH_CLUSTERS_PARQUET_PATH)
  
  # Extract features and clusters separately
  X = cand_embeddings_df.drop(['cand_id', 'cluster']).to_pandas()
  clusters = cand_embeddings_df['cluster'].to_numpy()
  cand_median_donations_df = (pl
                              .scan_parquet(constants.CONTRIBUTIONS_TABLE_PARQUET_PATH)
                              .group_by('candidate.id')
                              .median()
                              .select(pl.col('candidate.id').alias('cand_id'), pl.col('amount').alias('median_donation_amount'))
                              .collect())
  median_donation_amounts = cand_embeddings_df.join(cand_median_donations_df, on='cand_id', how='left', maintain_order='left')['median_donation_amount'].log()
  candidate_ico_df = pl.scan_parquet(constants.CONTRIBUTIONS_TABLE_PARQUET_PATH).group_by('candidate.id').first().select([pl.col('candidate.id').alias('cand_id'), pl.col('ico.status')]).collect()
  candidate_icos = cand_embeddings_df.join(candidate_ico_df, on='cand_id', how='left', maintain_order='left')['ico.status']

  reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    n_components=2,
    random_state=10,
  )
  reduced = reducer.fit_transform(X)

  # Plot embeddings reduced to 2D with cluster colors
  plt.figure(figsize=(10,6))
  
  # Create discrete colors for each cluster
  unique_clusters = sorted(set(clusters))
  colors = plt.cm.tab10(range(len(unique_clusters)))
  
  # Plot each cluster separately for discrete legend
  for i, cluster in enumerate(unique_clusters):
    mask = clusters == cluster
    plt.scatter(reduced[mask, 0], reduced[mask, 1], 
               c=[colors[i]], s=8, alpha=0.7, label=f'Cluster {cluster}')
  
  plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.title('UMAP of candidate embeddings colored by K-means clusters')
  plt.xlabel('UMAP-1'); plt.ylabel('UMAP-2')
  
  # Save the plot
  plt.tight_layout()
  plt.savefig('./images/candidate_embedding_umap.png', dpi=150, bbox_inches='tight')
  plt.show()


if __name__ == '__main__':
  main()
