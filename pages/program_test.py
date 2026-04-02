import streamlit as st
# from similarity import similarity_engine, hybrid_engine, mmr_recommendations
import pandas as pd

#####################
## Main.py content ##
#####################

import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def user_profile_def(user_id, view_history, programs_tfidf):
    view_history = pd.read_csv("data/view_history.csv")
    view_history_user = view_history[view_history["user_id"] == user_id]
    programs_tfidf = pd.read_csv("data/programs_tfidf.csv")

    user_profile = pd.merge(view_history_user, programs_tfidf, on="program_id", how="left")
    user_profile.dropna(axis=0)

    tfidf_cols = user_profile.columns[user_profile.columns.get_loc('save') + 1:]

    user_profile[tfidf_cols] = user_profile[tfidf_cols].multiply(user_profile['listen_ratio'], axis=0)

    user_profile = user_profile.groupby('user_id')[tfidf_cols].mean().reset_index()
    return user_profile


def cosine_similarity_def(user_profile, programs_tfidf):
    user_profile = pd.DataFrame(user_profile)
    tfidf_cols = [col for col in user_profile.columns if col != 'user_id']
    user_vectors = user_profile[tfidf_cols].values
    program_vectors = programs_tfidf[tfidf_cols].values

    sim_matrix = cosine_similarity(user_vectors, program_vectors)
     
    sim_df = pd.DataFrame(
        sim_matrix,
        index=user_profile['user_id'],
        columns=programs_tfidf['program_id']
    )

    return sim_df


def top_50_recommendations(user_id, similarity_df, threshold_amount=30):
    # Adjust this threshold to whatever we need for the exposure fairness and diversity/mmr
    similarity_df = similarity_df.transpose().sort_values(by=user_id, ascending=False)
    similarity_df = pd.merge(similarity_df, pd.read_csv("data/programs.csv"), on="program_id", how="left").dropna()
    return similarity_df.iloc[:threshold_amount]


def exposure_fairness(top_50_recommendations):
    return top_50_recommendations.sort_values(by="inclusion_score", ascending=False).reset_index(drop=True)


def recommendation_content(user_id, view_history, programs_tfidf):
    user_profile = user_profile_def(user_id, view_history, programs_tfidf)
    
    similarity_df = cosine_similarity_def(user_profile, programs_tfidf)

    top50_programs = top_50_recommendations(user_id, similarity_df)

    exposure = exposure_fairness(top50_programs)

    return top50_programs

os.chdir("/home/anass/university/msc_applied_data_science/INFOMPPM/INFOMPPM")

#####################
## End             ##
#####################


# Data loading functions with caching
# @st.cache_data
# def load_data():
#     df_synth_simple = pd.read_csv('synthetic_view_history_simple.csv')

#     files = ['data/documentaries.pkl', 'data/comedy.pkl',
#              'data/sports.pkl', 'data/science-and-nature.pkl']
#     df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
#     df = df.dropna(subset=['synopsis_large']).reset_index(drop=True)

#     return df, df_synth_simple

# @st.cache_data
# def load_similarity(df):
#     return similarity_engine(df)

# @st.cache_data
# def load_hybrid(_df, _user_df, _content_matrix, alpha):
#     return hybrid_engine(_df, _user_df, _content_matrix, alpha)



# Main Streamlit app
# df, user_df = load_data()
# content_matrix, tfidf_matrix, tfidf = load_similarity(df)

st.title("Recommendation engine")

user_df = pd.read_csv("data/view_history.csv")
view_history = pd.read_csv("data/view_history.csv")

if "view_history" not in st.session_state:
    st.session_state["view_history"] = view_history

programs_tfidf = pd.read_csv("data/programs_tfidf.csv")

# User interface for selecting user and recommendation style
user_id = st.selectbox("Select user", user_df['user_id'].unique())
alpha = st.select_slider(
    "Recommendation style",
    options=["More personalized", "Balanced", "More diverse"],
    value="Balanced"
)

# Map slider choice to alpha value for blending
alpha_map = {"More personalized": 0.2, "Balanced": 0.5, "More diverse": 0.8}
alpha_val = alpha_map[alpha]

# Load the hybrid similarity matrix based on user selection
# collab_matrix, hybrid_matrix = load_hybrid(df, user_df, content_matrix, alpha_val)


# Placeholder functions for user interactions
def on_play(user_id, program_id, view_history, save_to_file=True):
    new_row = pd.DataFrame([{
        "user_id": user_id,
        "program_id": program_id,
        "listen_ratio": 1.0,
        "save": "no"
    }])

    if save_to_file:
        new_row.to_csv("data/view_history.csv", mode="a", header=False, index=False)

    view_history = pd.concat([st.session_state["view_history"], new_row], ignore_index=True)
    return view_history

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


# Function to display recommendations in Streamlit
def display_recommendations(results, user_id):
    for _, row in results.iterrows():
        save_key = f"save_{row['title']}"
        if save_key not in st.session_state:
            st.session_state[save_key] = False

        if ':' in row['title']:
            show_name, episode_name = row['title'].split(':', 1)
        else:
            show_name = row['title']
            episode_name = None

        col1, col2 = st.columns([1, 2])
        with col1:
            if pd.notna(row['image']):
                st.image(row['image'], use_container_width=True)
        with col2:
            st.markdown(f"### {show_name.strip()}")
            if episode_name:
                st.markdown(f"*{episode_name.strip()}*")
            st.caption(f"📂 {row['category']}  ·  ⏱ {row['duration_txt']}")
            st.write(row['synopsis_small'])

            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                if st.button("▶ Play", key=f"play_{row['program_id']}", use_container_width=True):
                    st.session_state["view_history"] = on_play(
                        user_id, row['program_id'], st.session_state["view_history"])
                    print(st.session_state["view_history"].tail(1))
                    st.toast(f"Playing {show_name.strip()}...")
            with btn_col2:
                label = "✅ Saved" if st.session_state[save_key] else "🔖 Save"
                if st.button(label, key=f"btn_{row['title']}", use_container_width=True):
                    st.session_state[save_key] = not st.session_state[save_key]
                    on_save(user_id, row['title'], st.session_state[save_key])  # TODO: implement
                    st.rerun()

        st.divider()

# Generate and display recommendations based on user input
results = recommendation_content(user_id, pd.read_csv("data/view_history.csv"), programs_tfidf)

if isinstance(results, str):
    st.warning(results)
else:
    display_recommendations(results, user_id)
