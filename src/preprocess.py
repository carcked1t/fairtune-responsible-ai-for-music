import pandas as pd
from src.config import PROCESSED_TRACKS
from src.data_loader import load_tracks

def preprocess_tracks():
    df = load_tracks()

    keep = [
        "track_id", "track_name", "artists", "album_name",
        "track_genre", "popularity", "duration_ms", "explicit",
        "danceability", "energy", "key", "loudness", "mode",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo", "time_signature"
    ]

    df = df[keep].copy()

    num_cols = [
        "popularity", "duration_ms", "danceability", "energy", "key",
        "loudness", "mode", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo",
        "time_signature"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=num_cols)

    df.to_parquet(PROCESSED_TRACKS, index=False)
    return df

def load_or_build_tracks():
    if PROCESSED_TRACKS.exists():
        return pd.read_parquet(PROCESSED_TRACKS)
    return preprocess_tracks()
