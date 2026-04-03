import streamlit as st
from similarity import similarity_engine, hybrid_engine, mmr_recommendations
import pandas as pd
from template import display_recommendations

# Data loading functions with caching
@st.cache_data
def load_data():
    df_synth_simple = pd.read_csv('synthetic_view_history_simple.csv')

    files = [
    'data/documentaries.pkl',
    'data/comedy.pkl',
    'data/sports.pkl',
    'data/science-and-nature.pkl',
    'data/arts.pkl',
    'data/cbbc.pkl',
    'data/entertainment.pkl',
    'data/films.pkl',
    'data/from-the-archives.pkl',
    'data/history.pkl',
    'data/lifestyle.pkl',
    'data/music.pkl',
    'data/signed.pkl']
    df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    df = df.dropna(subset=['synopsis_large']).reset_index(drop=True)

    return df, df_synth_simple

@st.cache_data
def load_similarity(df):
    return similarity_engine(df)

@st.cache_data
def load_hybrid(_df, _user_df, _content_matrix, alpha):
    return hybrid_engine(_df, _user_df, _content_matrix, alpha)


# Main Streamlit app
df, user_df = load_data()
content_matrix, tfidf_matrix, tfidf = load_similarity(df)

st.title("Your Personalized Recommendations")

# User interface for selecting user and recommendation style
user_id = st.selectbox("Select user", user_df['user_id'].unique())
alpha = st.select_slider(
    "Recommendation style",
    options=["More personalized", "Balanced", "More diverse"],
    value="Balanced"
)

# Map slider choice to alpha value for blending
lambda_map = {"More personalized": 0.2, "Balanced": 0.5, "More diverse": 0.8}
lambda_val = lambda_map[alpha]

# Load the hybrid similarity matrix based on user selection
collab_matrix, hybrid_matrix = load_hybrid(df, user_df, content_matrix, lambda_val)


# Placeholder functions for user interactions
def on_play(user_id, title):
    """
    Called when a user clicks Play.
    Should add a watch event to the user's view history.

    TODO:
    - Append a row to synthetic_view_history_simple.csv
      with user_id, title, save=no, timestamp
    - Or write to a database
    """
    pass

# Placeholder function for Save button
def on_save(user_id, title, saved: bool):
    """
    Called when a user toggles Save.
    Should update the save status in the user's view history.

    TODO:
    - If saved=True: add or update row with save=yes
    - If saved=False: remove or update row with save=no
    - Or write to a database
    """
    pass


# Generate and display recommendations based on user input
results = mmr_recommendations(user_id, user_df, df, hybrid_matrix, content_matrix)
if isinstance(results, str):
    st.warning(results)
else:
    display_recommendations(results, user_id)