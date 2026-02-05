import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

FEATURE_COLS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "time_signature"
]

def build_feature_matrix(tracks_df):
    X = tracks_df[FEATURE_COLS].values.astype(np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler

def recommend_from_seed_playlist(tracks_df, seed_track_ids, top_n=100):
    """
    Simulates a user profile from seed tracks:
    - user vector = average audio features of seed tracks
    - recommend closest tracks by cosine similarity
    """
    df = tracks_df.copy()

    Xs, _ = build_feature_matrix(df)

    # seed indices
    seed_mask = df["track_id"].isin(seed_track_ids)
    if seed_mask.sum() == 0:
        raise ValueError("None of the seed_track_ids exist in the dataset.")

    seed_vec = Xs[seed_mask.values].mean(axis=0, keepdims=True)

    sims = cosine_similarity(seed_vec, Xs).flatten()

    # Remove seed tracks from recs
    sims[seed_mask.values] = -1

    top_idx = np.argsort(sims)[::-1][:top_n]

    recs = df.iloc[top_idx][
        ["track_id", "track_name", "artists", "album_name", "track_genre", "popularity"]
    ].copy()

    recs["relevance_score"] = sims[top_idx]
    recs = recs.rename(columns={"track_genre": "genre", "artists": "artist_name"})
    recs["artist_popularity"] = recs["popularity"]

    return recs
