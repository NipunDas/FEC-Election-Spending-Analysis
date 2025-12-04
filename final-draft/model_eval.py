"""
A script that evaluates all models that predict contributions.
"""
import numpy as np
from data_loader import DataLoader
from models.random import RandomModel
from models.heuristic import HeuristicModel
from models.cf import CFModel
from models.als import ALSModel

TOP_K = 20

def precision_at_k(recommended, relevant, k):
  hits = sum(1 for c_id in recommended[:k] if c_id in relevant)
  return hits/k

def recall_at_k(recommended, relevant, k):
  hits = sum(1 for c_id in recommended[:k] if c_id in relevant)
  return hits/len(relevant)

def average_precision_at_k(recommended, relevant, k):
  average_precision, hits = 0.0, 0
  for i, c_id in enumerate(recommended[:k], start=1):
    if c_id in relevant:
      hits += 1
      average_precision += hits/i
  return average_precision / min(len(relevant), k)

def evaluate_rankings(rec_lists, relevant_committee_dict, k):
  p_list, r_list, ap_list = [], [], []

  for cand_id, recs in rec_lists.items():
    rel = relevant_committee_dict[cand_id]
    p_list.append(precision_at_k(recs, rel, k))
    r_list.append(recall_at_k(recs, rel, k))
    ap_list.append(average_precision_at_k(recs, rel, k))

  return (
    np.mean(p_list),
    np.mean(r_list),
    np.mean(ap_list),
  )

def main():
  print('Loading and preparing data...')
  data_loader = DataLoader()
  data_loader.prepare_data()

  models = {
    'Random': RandomModel(),
    'Heuristic': HeuristicModel(),
    # 'Collaborative Filtering': CFModel(n_factors=15),
    'ALS': ALSModel(),
  }

  X_train, y_train = data_loader.get_training_data()

  for model_name, model in models.items():
    print(f'Training {model_name} model...')
    model.fit(X_train, y_train)

  predicted_donors = dict()
  for model_name, model in models.items():
    print(f'Generating predictions for {model_name} model...')

    predicted_donors[model_name] = dict()
    for cand_id in data_loader.get_eval_candidates():
      X_potential_contributions_df = data_loader.get_potential_contributions_df(cand_id)
      predicted_donors[model_name][cand_id] = model.recommend(X_potential_contributions_df)

  # Evaluate all models
  print(f'\n=== Model comparison (TOP_K = {TOP_K}) ===')
  for name in models.keys():
    p, r, map = evaluate_rankings(predicted_donors[name], data_loader.relevant_committees_per_candidate, TOP_K)
    print(f'{name} model:')
    print(f"  Precision@K: {p:.4f}")
    print(f"  Recall@K:    {r:.4f}")
    print(f"  MAP@K:       {map:.4f}")
    print()
    

if __name__ == '__main__':
  main()
