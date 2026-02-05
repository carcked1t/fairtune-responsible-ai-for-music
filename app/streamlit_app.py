import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocess import load_or_build_tracks
from src.baseline_content import recommend_from_seed_playlist
from src.reranker_bandit import epsilon_greedy_rerank
from src.evaluation import evaluate_playlist

st.set_page_config(page_title="FairTune", layout="wide")

st.title("FairTune: Responsible AI for Music Discovery")
st.caption("Audits recommendations for popularity bias & genre filter bubbles, then re-ranks using ε-greedy bandits.")

@st.cache_data
def load_data():
    return load_or_build_tracks()

tracks = load_data()

# Pick seed tracks
st.sidebar.header("Controls")

all_tracks = tracks[["track_id", "track_name", "artists", "track_genre"]].copy()
all_tracks["label"] = all_tracks["track_name"] + " — " + all_tracks["artists"]

seed_choices = st.sidebar.multiselect(
    "Pick 3–10 seed tracks (simulates user taste)",
    options=all_tracks["label"].tolist()[:2000], 
    default=all_tracks["label"].tolist()[0:5]
)

epsilon = st.sidebar.slider("Exploration rate ε", 0.0, 0.5, 0.2, 0.05)
top_n = st.sidebar.slider("Playlist size", 5, 30, 10, 1)

label_to_id = dict(zip(all_tracks["label"], all_tracks["track_id"]))
seed_ids = [label_to_id[x] for x in seed_choices if x in label_to_id]

# Generate baseline candidates
baseline = recommend_from_seed_playlist(tracks, seed_ids, top_n=100)
reranked = epsilon_greedy_rerank(baseline, k=top_n, epsilon=epsilon)

baseline_top = baseline.sort_values("relevance_score", ascending=False).head(top_n)

# Evaluate
global_threshold = tracks["popularity"].quantile(0.30)

m_base = evaluate_playlist(baseline_top, tracks)
m_rerank = evaluate_playlist(reranked, tracks)



col1, col2 = st.columns(2)

with col1:
    st.subheader("Baseline Playlist")
    st.dataframe(
        baseline_top[["track_name", "artist_name", "genre", "artist_popularity", "relevance_score"]],
        use_container_width=True
    )

with col2:
    st.subheader("FairTune Reranked Playlist")
    st.dataframe(
        reranked[["track_name", "artist_name", "genre", "artist_popularity", "relevance_score", "discovery_score"]],
        use_container_width=True
    )

st.divider()
st.subheader("Fairness Metrics Comparison")

metrics_df = pd.DataFrame([m_base, m_rerank], index=["Baseline", "FairTune"])
st.dataframe(metrics_df, use_container_width=True)

st.divider()
st.subheader("Fairness vs Accuracy Tradeoff (ε sweep)")

eps_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
entropy_vals, gini_vals, avg_rel = [], [], []

for e in eps_values:
    reranked = epsilon_greedy_rerank(
        baseline,
        k=top_n,
        epsilon=e,
        seed=42,
        catalog_popularity=tracks["popularity"].tolist(),
        catalog_genres=tracks["track_genre"].tolist()
    )
    m = evaluate_playlist(reranked, tracks)
    entropy_vals.append(m["entropy_genre"])
    gini_vals.append(m["gini_popularity"])
    avg_rel.append(reranked["relevance_score"].mean())

fig = plt.figure()
plt.plot(eps_values, entropy_vals, marker="o", label="Genre Entropy (↑ better)")
plt.plot(eps_values, gini_vals, marker="o", label="Popularity Gini (↓ better)")
plt.plot(eps_values, avg_rel, marker="o", label="Avg Relevance (↑ better)")
plt.xlabel("ε (exploration rate)")
plt.ylabel("Metric value")
plt.legend()
st.pyplot(fig)
