# model_chart.py
import matplotlib.pyplot as plt
import numpy as np

# ---- Data ----
models = ['Random', 'Heuristic']
metrics = ['Precision@K', 'Recall@K', 'MAP@K']
values = {
    'Random':    [0.0075, 0.0068, 0.0019],
    'Heuristic': [0.1280, 0.1439, 0.0702],
}

# ---- Plot ----
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))
bars1 = plt.bar(x - width/2, values['Random'], width, label='Random')
bars2 = plt.bar(x + width/2, values['Heuristic'], width, label='Heuristic')

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

plt.tight_layout()
plt.savefig('./images/model_comparison_topk20.png', dpi=150, bbox_inches='tight')
plt.show()
