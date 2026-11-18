"""Exactly reconstruct the 660-sample Caltech101-20 subset created on 2026-10-03.
Usage: python Caltech10120_balanced_selection_seed42.py [original.mat] [output.mat]
The ordering is as selected by a single NumPy PCG64 Generator, seed=42,
used sequentially for the 20 sorted original class labels.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.io import loadmat, savemat

source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('Caltech101-20.mat')
target = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('Caltech101-20_balanced_33perclass_reproduced.mat')
original = loadmat(source)
y = np.asarray(original['Y']).ravel()
rng = np.random.default_rng(42)
indices = np.concatenate([
    rng.choice(np.flatnonzero(y == label), size=33, replace=False)
    for label in np.unique(y)
])
assert indices.shape == (660,)
subset = {key: value for key, value in original.items() if not key.startswith('__')}
subset['Y'] = np.asarray(original['Y'])[indices, :]
X = np.asarray(original['X'], dtype=object)
subset['X'] = np.empty_like(X)
for j in range(X.size):
    features = X.flat[j]
    subset['X'].flat[j] = features[indices, :]
subset['lenSmp'] = np.full_like(original['lenSmp'], 33)
savemat(target, subset, do_compression=True)
print('Saved:', target, '| rows:', len(indices), '| labels:', np.unique(subset['Y'], return_counts=True))
