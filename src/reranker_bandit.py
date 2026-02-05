import numpy as np
import pandas as pd


def compute_discovery_score(
    df: pd.DataFrame,
    catalog_popularity=None,
    catalog_genres=None,
    target_popularity=None
):
    """
    Adds:
      - genre_rarity
      - long_tail_bonus
      - pop_balance_penalty
      - discovery_score

    df must contain:
      - track_id
      - genre
      - artist_popularity
      - relevance_score

    catalog_popularity: list/np.array/Series of popularity values from FULL dataset
    catalog_genres: list/np.array/Series of genres from FULL dataset
    """

    out = df.copy()

    # 1) Genre rarity (prefer rare)
    if catalog_genres is None:
        # fallback: rarity within candidates
        genre_counts = out["genre"].value_counts()
    else:
        catalog_genres = pd.Series(catalog_genres).dropna()
        genre_counts = catalog_genres.value_counts()

    out["genre_rarity"] = out["genre"].map(lambda g: 1.0 / (float(genre_counts.get(g, 0)) + 1e-9))

    # 2) Long-tail bonus (global normalized)
    pop = out["artist_popularity"].astype(float)

    if catalog_popularity is None:
        # fallback: normalize within candidates (old behavior)
        global_min = float(pop.min())
        global_max = float(pop.max())
    else:
        catalog_popularity = np.array(catalog_popularity, dtype=float)
        catalog_popularity = catalog_popularity[~np.isnan(catalog_popularity)]

        if len(catalog_popularity) == 0:
            global_min = float(pop.min())
            global_max = float(pop.max())
        else:
            global_min = float(np.min(catalog_popularity))
            global_max = float(np.max(catalog_popularity))

    pop_norm = (pop - global_min) / (global_max - global_min + 1e-9)
    pop_norm = np.clip(pop_norm, 0.0, 1.0)

    # 1.0 = very tail, 0.0 = very head
    out["long_tail_bonus"] = 1.0 - pop_norm

    # 3) Popularity balance penalty (avoid extreme picks)
    # If we always push tail, we might destroy relevance.
    # If we always push head, we get popularity bias.
    # So we penalize deviation from a target popularity.
    # BEST target: median popularity of the CANDIDATES (stable).

    if target_popularity is None:
        target_popularity = float(pop.median())

    pop_dev = np.abs(pop - target_popularity)

    # normalize deviation to [0,1]
    if pop_dev.max() - pop_dev.min() < 1e-9:
        out["pop_balance_penalty"] = 0.0
    else:
        out["pop_balance_penalty"] = (pop_dev - pop_dev.min()) / (pop_dev.max() - pop_dev.min() + 1e-9)

    # 4) Final discovery score
    # genre_rarity: higher is better
    # long_tail_bonus: higher is better
    # pop_balance_penalty: lower is better
    
    out["discovery_score"] = (
        0.45 * out["genre_rarity"]
        + 0.35 * out["long_tail_bonus"]
        - 0.20 * out["pop_balance_penalty"]
    )

    return out


def epsilon_greedy_rerank(
    candidates_df: pd.DataFrame,
    k: int = 10,
    epsilon: float = 0.2,
    seed: int = 42,
    catalog_popularity=None,
    catalog_genres=None
):
    """
    ε-greedy reranking:
      - With probability ε: explore (pick highest discovery_score)
      - With probability 1-ε: exploit (pick highest relevance_score)

    IMPORTANT:
    We compute discovery_score ONCE on the candidate set,
    then select without replacement.
    """

    rng = np.random.default_rng(seed)

    # target popularity = median of candidates
    target_popularity = float(candidates_df["artist_popularity"].median())

    df = compute_discovery_score(
        candidates_df,
        catalog_popularity=catalog_popularity,
        catalog_genres=catalog_genres,
        target_popularity=target_popularity
    )

    selected = []
    used = set()

    for _ in range(k):
        left = df[~df["track_id"].isin(used)]
        if left.empty:
            break

        explore = rng.random() < epsilon

        if explore:
            # explore = sample from top discovery candidates
            pool = left.sort_values("discovery_score", ascending=False).head(10)
            pick = pool.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        else:
            #best relevance
            pick = left.sort_values("relevance_score", ascending=False).iloc[0]

        selected.append(pick)
        used.add(pick["track_id"])

    return pd.DataFrame(selected)
