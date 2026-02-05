# FairTune   
### Responsible AI for Music Discovery (Popularity Bias + Genre Diversity)

FairTune is a Streamlit dashboard that audits a content-based music recommender for **popularity bias** and **genre filter bubbles**, then applies a lightweight **ε-greedy bandit re-ranking** strategy to improve fairness while maintaining relevance.

This project demonstrates how a recommender can become more responsible by explicitly balancing:
- **Relevance** (recommend what the user likes)
- **Discovery** (introduce novelty and diversity)
- **Fair Exposure** (avoid only recommending the most popular artists)


##  What FairTune Does

FairTune runs two recommendation pipelines side-by-side:

### 1) Baseline recommender (content-based)
- Takes 3–10 seed tracks (simulated user taste)
- Builds a candidate list based on similarity
- Returns the top-N ranked purely by relevance

### 2) FairTune reranker (ε-greedy bandit)
- Starts from the same candidate list
- Computes a **discovery score** for each track
- Uses ε-greedy selection:
  - with probability **ε** → explore (pick best discovery)
  - with probability **1 − ε** → exploit (pick best relevance)

This produces a playlist that is still aligned with user taste but improves fairness signals.


##  Dashboard Controls (Parameters)

###  1) Seed Tracks (3–10)
**What it is:**  
The tracks you pick represent the user's listening taste.

**Why it matters:**  
More seed tracks = more stable personalization  
Fewer seed tracks = noisier personalization (more diversity possible)

**Expected behavior:**
- With 3 seeds → higher variety, slightly lower relevance
- With 10 seeds → stronger relevance, lower exploration


###  2) Exploration Rate ε (0.0 → 0.5)
**What it is:**  
Controls how often the system chooses a discovery-focused track instead of the most relevant one.

**Interpretation:**
- **ε = 0.0** → Pure relevance ranking (baseline behavior)
- **ε = 0.1–0.3** → Balanced (recommended range)
- **ε = 0.4–0.5** → Strong exploration (fairness improves, relevance may drop)

**Expected behavior:**
- Higher ε should improve:
  - genre diversity (entropy)
  - long-tail exposure (more niche artists)
- But may slightly reduce:
  - average relevance score


###  3) Playlist Size (5 → 30)
**What it is:**  
Number of tracks shown in the final playlist.

**Why it matters:**
- Small playlists (5–10) are harder to make fair because every track matters.
- Larger playlists (20–30) allow more diversity without losing relevance.


##  What “Fair / Responsible” Means Here

FairTune focuses on **exposure fairness**, not demographic fairness.

The system is considered more responsible if it:
- avoids recommending only extremely popular artists
- increases exposure to long-tail (less popular) artists
- avoids genre collapse (same genre repeated)
- maintains reasonable relevance to user taste


##  Metrics Explained (What to Look For)

FairTune compares Baseline vs FairTune using three fairness metrics:


### 1) Popularity Gini (↓ lower is better)
**What it measures:**  
How unequal the popularity distribution is inside the playlist.

**Interpretation:**
- **Gini ≈ 0.0** → popularity is evenly spread  
- **Gini ≈ 1.0** → playlist is dominated by a few very popular artists  

**What values mean:**
- 0.2–0.4 → healthy spread
- 0.5–0.7 → strong popularity bias
- 0.7+ → extreme head-only playlist

**FairTune goal:**  
Reduce popularity Gini without destroying relevance.


### 2) Genre Entropy (↑ higher is better)
**What it measures:**  
How diverse the genres are in the playlist.

**Interpretation:**
- Low entropy → genre filter bubble  
- High entropy → more variety and exploration  

**What values mean:**
- ~1.0 → mostly one genre  
- ~2.0–3.5 → diverse playlist  
- 3.5+ → very diverse (depends on dataset size)

**FairTune goal:**  
Increase entropy without recommending irrelevant genres.


### 3) Long-Tail Exposure (↑ higher is better)
**What it measures:**  
What fraction of playlist tracks come from the bottom popularity region of the full catalog.

**How it is computed in code:**
- Compute the 30th percentile popularity threshold using the entire dataset
- A track is "long-tail" if popularity ≤ threshold
- Long-tail exposure = fraction of playlist in that group

**Interpretation:**
- 0.0 → no long-tail artists recommended
- 0.3 → 30% of playlist is long-tail
- 0.5+ → very discovery-heavy playlist

**FairTune goal:**  
Increase long-tail exposure while keeping relevance stable.


##  Additional Internal Scores (Used in Re-ranking)

### Relevance Score (↑ higher is better)
This comes from the baseline recommender.

It represents how similar a candidate track is to the seed tracks.

FairTune tries not to destroy this.


### Discovery Score (↑ higher is better)
This is computed inside `src/reranker_bandit.py`.

It is designed to promote:
- rare genres
- low-popularity artists
- but avoid extreme popularity deviation

Discovery score = weighted combination of:

#### 1) Genre Rarity Bonus
Tracks in rarer genres get boosted.

#### 2) Long Tail Bonus
Tracks from lower popularity artists get boosted.

#### 3) Popularity Balance Penalty
Prevents the system from only picking extreme outliers.


##  What the Reranker is Actually Doing

### File: `src/reranker_bandit.py`

The reranker does NOT change the baseline model.

Instead, it re-orders the baseline candidate list.

#### Step-by-step:
1. Baseline produces top 100 candidates ranked by relevance.
2. FairTune computes discovery_score for each candidate.
3. Playlist is built iteratively:
   - with probability ε → pick best discovery track
   - otherwise → pick best relevance track
4. Tracks are selected without replacement.

This is why the playlist is still personalized, but fairness improves.


##  Fairness vs Accuracy Tradeoff Graph (ε sweep)

The dashboard runs the reranker for ε values:
`[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]`

For each ε, it plots:
- Genre entropy (↑)
- Popularity gini (↓)
- Avg relevance (↑)

This shows the real tradeoff:
- More exploration increases fairness
- But can slightly reduce relevance


##  Project Structure
```text
fairtune-resp-ai-for-music/
│
├── app/
│ └── streamlit_app.py # Streamlit UI + plots
│
├── data/
│ └── tracks.csv # Input dataset
│
├── src/
│ ├── preprocess.py # Cleaning + schema normalization
│ ├── data_loader.py # Loads dataset from /data
│ ├── baseline_content.py # Baseline recommender
│ ├── reranker_bandit.py # ε-greedy fairness reranker
│ ├── metrics.py # Gini, entropy, long-tail exposure
│ ├── evaluation.py # Computes metrics for a playlist
│ └── config.py # Constants (if used)
│
├── requirements.txt
└── README.md
```


##  How to Run

### 1) Install dependencies
pip install -r requirements.txt

### 2) Run Streamlit
streamlit run app/streamlit_app.py

## Expected Results (What “Better” Looks Like)

A more responsible playlist typically shows:

- Higher genre entropy
- Higher long-tail exposure
- Lower popularity gini
- Slightly lower relevance (acceptable tradeoff)

FairTune is not trying to maximize fairness at all costs.
It aims for a realistic middle ground where personalization still holds.

## Future Improvements

True contextual bandits (learning user feedback)

More robust long-tail definition (artist-level instead of track-level)

Constraint-based re-ranking (hard limits on head exposure)

Add novelty and serendipity metrics

Multi-objective optimization instead of ε-greedy

## Summary

FairTune is a practical demo of how recommender systems can be audited and improved for responsible AI goals using transparent fairness metrics, controllable exploration, lightweight re-ranking.

It is designed for interpretability and experimentation rather than black-box modeling.
