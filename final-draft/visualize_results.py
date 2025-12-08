# model_chart.py
import matplotlib.pyplot as plt
import numpy as np

# ---- Data ----
models = ['Random', 'Heuristic', 'Collaborative Filtering', 'Random Forest']
metrics = ['Precision@K', 'Recall@K', 'MAP@K']
values = {
    'Random':                  [0.0016, 0.0006, 0.0003],
    'Heuristic':               [0.0457, 0.0331, 0.0270],
    'SVD':                     [0.0019, 0.0007, 0.0003],
    'ALS':                     [0.1005, 0.1956, 0.1353],
    'Random Forest':           [0.0034, 0.0051, 0.0010],
}

# ---- Plot ----
x = np.arange(len(metrics))
width = 0.15
offset = width * 2

plt.figure(figsize=(12, 6))
bars1 = plt.bar(x - offset, values['Random'], width, label='Random')
bars2 = plt.bar(x - width, values['Heuristic'], width, label='Heuristic')
bars3 = plt.bar(x, values['SVD'], width, label='SVD')
bars4 = plt.bar(x + width, values['ALS'], width, label='ALS')
bars5 = plt.bar(x + offset, values['Random Forest'], width, label='Random Forest')

plt.title('Model comparison (TOP_K = 20)')
plt.ylabel('Score')
plt.xticks(x, metrics)
plt.legend()
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# annotate bars with values
def annotate(bars):
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h, f'{h:.4f}',
                 ha='center', va='bottom', fontsize=9)

annotate(bars1)
annotate(bars2)
annotate(bars3)
annotate(bars4)
annotate(bars5)

plt.tight_layout()
plt.savefig('./images/model_comparison_topk20.png', dpi=150, bbox_inches='tight')
plt.show()
