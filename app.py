import streamlit as st
from similarity import similarity_engine, hybrid_engine, mmr_recommendations
import pandas as pd

# Data loading functions with caching
@st.cache_data
def load_data():
    df_synth_simple = pd.read_csv('synthetic_view_history_simple.csv')

    files = ['data/documentaries.pkl', 'data/comedy.pkl',
             'data/sports.pkl', 'data/science-and-nature.pkl']
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

st.title("Recommendation engine")

user_id = st.selectbox("Select user", user_df['user_id'].unique())
alpha = st.select_slider(
    "Recommendation style",
    options=["More personalized", "Balanced", "More diverse"],
    value="Balanced"
)

alpha_map = {"More personalized": 0.2, "Balanced": 0.5, "More diverse": 0.8}
alpha_val = alpha_map[alpha]

collab_matrix, hybrid_matrix = load_hybrid(df, user_df, content_matrix, alpha_val)

if st.button("Get recommendations"):
    results = mmr_recommendations(user_id, user_df, df, hybrid_matrix, content_matrix)
    st.write(results)
