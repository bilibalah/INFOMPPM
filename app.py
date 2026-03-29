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



def display_recommendations(results):
    for _, row in results.iterrows():
        col1, col2 = st.columns([1, 3])

        save_key = f"save_{row['title']}"
        if save_key not in st.session_state:
            st.session_state[save_key] = False

        with col1:
            if pd.notna(row['image']):
                st.image(row['image'], use_container_width=True)
        with col2:
            st.markdown(f"### {row['title']}")

            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                st.caption(f"📂 {row['category']}")
            with meta_col2:
                st.caption(f"⏱ {row['duration_txt']}")

            st.write(row['synopsis_small'])

            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                st.button("▶ Play", key=f"play_{row['title']}", use_container_width=True)
            with btn_col2:
                label = "✅ Saved" if st.session_state[save_key] else "🔖 Save"
                if st.button(label, key=f"btn_{row['title']}", use_container_width=True):
                    st.session_state[save_key] = not st.session_state[save_key]
                    st.rerun()

        st.divider()

results = mmr_recommendations(user_id, user_df, df, hybrid_matrix, content_matrix)
if isinstance(results, str):
    st.warning(results)
else:
    display_recommendations(results)