import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
fig, ax = plt.subplots()
found = False
for name in ('baseline', 'augmented'):
    path = ROOT / 'results' / name / 'history.csv'
    if not path.is_file():
        continue
    with path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    ax.plot([int(r['epoch']) for r in rows], [float(r['val_accuracy']) for r in rows], marker='o', label=name)
    found = True
if not found:
    raise SystemExit('No histories found; train models first')
ax.set(xlabel='Epoch', ylabel='Validation accuracy', title='MNIST CNN validation accuracy')
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
out = ROOT / 'results' / 'accuracy.png'
fig.savefig(out, dpi=180)
print('Saved:', out)
