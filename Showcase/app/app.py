import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data():
    """Load all data fresh (no cache so play/save updates are immediate)."""
    view_history = pd.read_csv("data/view_history.csv")
    programs = pd.read_csv("data/programs.csv")
    programs_tfidf = pd.read_csv("data/programs_tfidf.csv")
    return view_history, programs, programs_tfidf


def append_to_view_history(user_id, program_id, save_flag):
    """Append a play/save interaction to view_history.csv."""
    filepath = "data/view_history.csv"
    existing = pd.read_csv(filepath)

    new_row = {col: None for col in existing.columns}
    new_row["user_id"] = user_id
    new_row["program_id"] = program_id
    new_row["listen_ratio"] = 1
    new_row["save"] = save_flag

    updated = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
    updated.to_csv(filepath, index=False)


# ═══════════════════════════════════════════════════════════════════════
# INTERACTION MATRIX  (shared by both pipelines)
# ═══════════════════════════════════════════════════════════════════════

def build_interaction_matrix(view_history):
    """
    Build a user × program interaction matrix.
    Score = listen_ratio, doubled when save == 'yes'.
    If a user listened to the same program multiple times, take the max.
    """
    df = view_history.copy()
    df["score"] = df["listen_ratio"]
    df.loc[df["save"] == "yes", "score"] *= 2
    interaction = df.groupby(["user_id", "program_id"])["score"].max().reset_index()
    matrix = interaction.pivot(index="user_id", columns="program_id", values="score").fillna(0)
    return matrix


# ═══════════════════════════════════════════════════════════════════════
# 1) CONTENT-BASED FILTERING
# ═══════════════════════════════════════════════════════════════════════

def content_based_scores(user_id, view_history, programs, programs_tfidf, genres):
    """
    Build a TF-IDF user profile from listening history and compute
    cosine similarity to every program.
    Returns a Series: program_id → similarity score.
    """
    view_user = view_history[view_history["user_id"] == user_id].copy()
    view_user = pd.merge(
        view_user, programs[["program_id", "category"]], on="program_id", how="left"
    )

    # Fall back to all user genres if no match with selected genres
    if genres and len(view_user[view_user["category"].isin(genres)]) > 0:
        view_user = view_user[view_user["category"].isin(genres)]
    else:
        genres = view_user["category"].dropna().unique().tolist()

    view_user = view_user.drop("category", axis=1)

    # Merge with TF-IDF features
    profile = pd.merge(view_user, programs_tfidf, on="program_id", how="left").dropna()
    if profile.empty:
        return pd.Series(dtype=float)

    profile.loc[profile["save"] == "yes", "listen_ratio"] *= 2

    tfidf_cols = profile.columns[profile.columns.get_loc("save") + 1 :]
    profile[tfidf_cols] = profile[tfidf_cols].multiply(profile["listen_ratio"], axis=0)

    # Average into a single user vector
    user_vector = profile[tfidf_cols].mean().values.reshape(1, -1)

    # Similarity to every program
    program_vectors = programs_tfidf[tfidf_cols].values
    sims = cosine_similarity(user_vector, program_vectors).flatten()

    return pd.Series(sims, index=programs_tfidf["program_id"])


# ═══════════════════════════════════════════════════════════════════════
# 2) COLLABORATIVE FILTERING  (user-based)
# ═══════════════════════════════════════════════════════════════════════

def collaborative_scores(user_id, interaction_matrix, k=5):
    """
    User-based collaborative filtering.
    1. Compute cosine similarity between the target user and all others.
    2. Pick the top-k most similar users.
    3. Predict a score for every program as the weighted average of
       those neighbours' interactions.
    Returns a Series: program_id → predicted score.
    """
    if user_id not in interaction_matrix.index:
        return pd.Series(dtype=float)

    user_vec = interaction_matrix.loc[[user_id]].values
    all_vecs = interaction_matrix.values

    # User-user similarity
    sims = cosine_similarity(user_vec, all_vecs).flatten()
    sim_series = pd.Series(sims, index=interaction_matrix.index)
    sim_series = sim_series.drop(user_id, errors="ignore")

    # Top-k neighbours
    top_neighbours = sim_series.nlargest(k)
    if top_neighbours.sum() == 0:
        return pd.Series(0, index=interaction_matrix.columns)

    # Weighted average of neighbour ratings
    neighbour_matrix = interaction_matrix.loc[top_neighbours.index]
    weights = top_neighbours.values
    weighted_scores = neighbour_matrix.T.dot(weights) / weights.sum()

    return pd.Series(weighted_scores, index=interaction_matrix.columns)


# ═══════════════════════════════════════════════════════════════════════
# 3) HYBRID COMBINATION
# ═══════════════════════════════════════════════════════════════════════

def hybrid_scores(cb_scores, cf_scores, alpha=0.5):
    """
    Combine content-based and collaborative scores.
    alpha controls the mix:  0 = pure collaborative, 1 = pure content-based.
    Both inputs are normalised to [0, 1] before combining.
    """
    all_programs = set(cb_scores.index) | set(cf_scores.index)
    cb = cb_scores.reindex(all_programs, fill_value=0)
    cf = cf_scores.reindex(all_programs, fill_value=0)

    # Min-max normalisation
    def normalise(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx > mn else s * 0

    cb_norm = normalise(cb)
    cf_norm = normalise(cf)

    combined = alpha * cb_norm + (1 - alpha) * cf_norm
    return combined


# ═══════════════════════════════════════════════════════════════════════
# 4) EXPOSURE FAIRNESS
# ═══════════════════════════════════════════════════════════════════════

def apply_exposure_fairness(candidates_df):
    """Sort candidates by inclusion_score to promote fair exposure."""
    if "inclusion_score" not in candidates_df.columns:
        return candidates_df
    return candidates_df.sort_values("inclusion_score", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
# 5) MMR DIVERSITY RE-RANKING
# ═══════════════════════════════════════════════════════════════════════

def mmr_rerank(candidates_df, programs_tfidf, top_n=15, lambda_param=0.5):
    """
    Maximal Marginal Relevance re-ranking.
    lambda_param: higher → more diversity, lower → more relevance.
    Uses 'hybrid_score' as the relevance signal.
    """
    candidate_ids = candidates_df["program_id"].tolist()
    if not candidate_ids:
        return pd.DataFrame()

    # Content similarity between candidates
    cand_tfidf = programs_tfidf[programs_tfidf["program_id"].isin(candidate_ids)].set_index("program_id")
    cand_tfidf = cand_tfidf.reindex(candidate_ids)
    tfidf_cols = [c for c in cand_tfidf.columns if c != "user_id"]
    vectors = cand_tfidf[tfidf_cols].values
    sim_matrix = pd.DataFrame(
        cosine_similarity(vectors), index=candidate_ids, columns=candidate_ids
    )

    # Normalise hybrid_score to [0, 1]
    max_hs = candidates_df["hybrid_score"].max()
    max_hs = max_hs if max_hs > 0 else 1

    selected = [candidate_ids.pop(0)]

    while len(selected) < top_n and candidate_ids:
        best_mmr, best_id = -float("inf"), None

        for cid in candidate_ids:
            relevance = (
                candidates_df.loc[candidates_df["program_id"] == cid, "hybrid_score"].values[0]
                / max_hs
            )
            max_sim = sim_matrix.loc[cid, selected].max()
            mmr = ((1 - lambda_param) * relevance) - (lambda_param * max_sim)

            if mmr > best_mmr:
                best_mmr = mmr
                best_id = cid

        selected.append(best_id)
        candidate_ids.remove(best_id)

    final = candidates_df.set_index("program_id").loc[selected].reset_index()
    return final


# ═══════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def recommend(user_id, view_history, programs, programs_tfidf, genres,
              alpha, top_n, lambda_param, top_k_candidates=50):
    """
    End-to-end hybrid recommendation pipeline:
      1. Content-based scores
      2. Collaborative scores
      3. Hybrid combination (α)
      4. Filter to genres & remove already-seen programs
      5. Take top-k candidates
      6. Exposure fairness
      7. MMR diversity re-ranking (λ)
    """
    # ── Resolve genres ──
    if not genres:
        user_cats = pd.merge(
            view_history[view_history["user_id"] == user_id],
            programs, on="program_id", how="left"
        )["category"].dropna().unique().tolist()
        genres = user_cats if user_cats else programs["category"].dropna().unique().tolist()

    # ── Scores ──
    cb = content_based_scores(user_id, view_history, programs, programs_tfidf, genres)
    interaction_matrix = build_interaction_matrix(view_history)
    cf = collaborative_scores(user_id, interaction_matrix)
    combined = hybrid_scores(cb, cf, alpha=alpha)

    # ── Build candidate dataframe ──
    score_df = combined.reset_index()
    score_df.columns = ["program_id", "hybrid_score"]
    score_df = pd.merge(score_df, programs, on="program_id", how="left").dropna()

    # Remove already-seen programs
    seen = view_history[view_history["user_id"] == user_id]["program_id"].unique()
    score_df = score_df[~score_df["program_id"].isin(seen)]

    # Filter to selected genres
    score_df = score_df[score_df["category"].isin(genres)]

    # Sort by hybrid score, take top candidates
    score_df = score_df.sort_values("hybrid_score", ascending=False).head(top_k_candidates)

    # Exposure fairness
    score_df = apply_exposure_fairness(score_df)

    # MMR diversity re-ranking
    final = mmr_rerank(score_df, programs_tfidf, top_n=top_n, lambda_param=lambda_param)

    return final


# ═══════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Hybrid Recommender", layout="wide")
st.title("📺 Hybrid Recommendation System")
st.caption("Content-based + Collaborative filtering with diversity control")

view_history, programs, programs_tfidf = load_data()

# ── Sidebar ──
with st.sidebar:
    st.header("Settings")

    user_ids = sorted(view_history["user_id"].unique())
    user_id = st.selectbox("User ID", user_ids)

    all_genres = sorted(programs["category"].dropna().unique())
    genres = st.multiselect(
        "Genres", all_genres,
        help="Leave empty to use all genres from the user's history",
    )

    st.divider()

    alpha = st.slider(
        "Content ↔ Collaborative (α)",
        0.0, 1.0, 0.5, 0.05,
        help="0 = pure collaborative · 1 = pure content-based",
    )
    lambda_param = st.slider(
        "Diversity (λ)",
        0.0, 1.0, 0.5, 0.05,
        help="Higher → more diversity · Lower → more relevance",
    )
    top_n = st.slider("Number of recommendations", 5, 30, 15)

    run = st.button("Get Recommendations", type="primary", use_container_width=True)


# ── Compute on button press or after play/save ──
def compute_and_store():
    """Run the pipeline and store results in session state."""
    vh = load_data()[0]  # reload fresh view history
    results = recommend(
        user_id, vh, programs, programs_tfidf,
        genres, alpha, top_n, lambda_param,
    )
    st.session_state["results"] = results
    st.session_state["rec_user_id"] = user_id
    st.session_state["rec_alpha"] = alpha
    st.session_state["rec_lambda"] = lambda_param


if run:
    with st.spinner("Computing recommendations..."):
        try:
            compute_and_store()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()


# ── Display results ──
if "results" in st.session_state:
    results = st.session_state["results"]
    rec_user_id = st.session_state["rec_user_id"]

    if results.empty:
        st.warning("No recommendations found for this combination.")
    else:
        st.subheader(f"Recommendations for {rec_user_id}")

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Picks", len(results))
        if "category" in results.columns:
            m2.metric("Genres", results["category"].nunique())
        m3.metric("α (CB ↔ CF)", st.session_state.get("rec_alpha", alpha))
        m4.metric("λ (diversity)", st.session_state.get("rec_lambda", lambda_param))

        st.divider()

        # Card grid — 5 per row
        COLS = 5
        for row_start in range(0, len(results), COLS):
            cols = st.columns(COLS)
            for j, col in enumerate(cols):
                idx = row_start + j
                if idx >= len(results):
                    break

                rec = results.iloc[idx]
                pid = rec["program_id"]
                title = str(rec.get("title", pid))
                genre = str(rec.get("category", ""))
                h_score = rec.get("hybrid_score", None)
                image = str(rec.get("image", ""))

                with col:
                    if image and image.startswith("http"):
                        st.image(image, use_container_width=True)

                    st.markdown(f"**{title}**")
                    caption = f"_{genre}_"
                    if h_score is not None and not pd.isna(h_score):
                        caption += f" · :green[{h_score:.2f}]"
                    st.caption(caption)

                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("▶ Play", key=f"play_{pid}", use_container_width=True):
                            append_to_view_history(rec_user_id, pid, "no")
                            compute_and_store()
                            st.toast(f"▶ **{title}** played — recommendations updated")
                            st.rerun()
                    with b2:
                        if st.button("💾 Save", key=f"save_{pid}", use_container_width=True):
                            append_to_view_history(rec_user_id, pid, "yes")
                            compute_and_store()
                            st.toast(f"💾 **{title}** saved — recommendations updated")
                            st.rerun()

            st.divider()

        # Raw data
        with st.expander("View raw data"):
            show_cols = [
                c for c in results.columns
                if c in list(programs.columns) + ["hybrid_score", "lambda_setting"]
            ]
            st.dataframe(results[show_cols].reset_index(drop=True), use_container_width=True)

        with st.expander("View updated listening history"):
            fresh_vh = load_data()[0]
            user_vh = fresh_vh[fresh_vh["user_id"] == rec_user_id].sort_index(ascending=False)
            st.dataframe(user_vh.reset_index(drop=True), use_container_width=True)
else:
    st.info("Configure the settings in the sidebar and click **Get Recommendations** to start.")