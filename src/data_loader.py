import pandas as pd
from pathlib import Path

def load_tracks():
    ROOT = Path(__file__).resolve().parents[1]
    csv_path = ROOT / "data" / "tracks.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find tracks.csv at: {csv_path}\n"
            f"Make sure it exists inside the /data folder."
        )

    return pd.read_csv(csv_path)
