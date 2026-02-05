import numpy as np
from src.metrics import gini, shannon_entropy, long_tail_exposure

def evaluate_playlist(df, tracks_df):
    return {
        "gini_popularity": gini(df["artist_popularity"].tolist()),
        "entropy_genre": shannon_entropy(df["genre"].tolist()),
        "long_tail_exposure": long_tail_exposure(
            df["artist_popularity"].tolist(),
            tracks_df["popularity"].tolist(),
            30
        )
    }
