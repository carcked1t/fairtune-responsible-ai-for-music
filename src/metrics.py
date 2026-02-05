import numpy as np
from collections import Counter

def gini(values):
    x = np.array(values, dtype=np.float64)
    x = x[x >= 0]
    if len(x) == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0

    x_sorted = np.sort(x)
    n = len(x_sorted)
    cumx = np.cumsum(x_sorted)

    g = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(g)

def shannon_entropy(categories):
    counts = Counter(categories)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts.values()], dtype=np.float64)
    return float(-np.sum(probs * np.log2(probs + 1e-12)))

def long_tail_exposure(playlist_popularity, catalog_popularity, threshold_percentile=30):
    playlist_popularity = np.array(playlist_popularity, dtype=np.float64)
    catalog_popularity = np.array(catalog_popularity, dtype=np.float64)

    if len(playlist_popularity) == 0 or len(catalog_popularity) == 0:
        return 0.0

    thresh = np.percentile(catalog_popularity, threshold_percentile)

    # rank weights
    ranks = np.arange(1, len(playlist_popularity) + 1)
    weights = 1.0 / np.log2(ranks + 1)

    tail_mask = (playlist_popularity <= thresh).astype(float)

    # weighted exposure in [0,1]
    return float(np.sum(weights * tail_mask) / (np.sum(weights) + 1e-12))
