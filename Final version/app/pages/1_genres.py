import streamlit as st
import pandas as pd
from recommendations import recommendation_content

@st.cache_data
def load_recommendation_data():
    view_history = pd.read_csv("../data/view_history.csv")
    programs = pd.read_csv("../data/programs.csv")
    programs_tfidf = pd.read_csv("../data/programs_tfidf.csv")
    return view_history, programs, programs_tfidf

def get_genre_recommendations(user_id, selected_genres, lambda_param=0.5):
    view_history, programs, programs_tfidf = load_recommendation_data()
    results = recommendation_content(user_id, view_history, programs, programs_tfidf, selected_genres, lambda_param)
    return results  # no extra merge needed, recommendation_content already handles it


def display_recommendations(results):
    for _, row in results.iterrows():
        save_key = f"save_{row['title']}"
        if save_key not in st.session_state:
            st.session_state[save_key] = False

        if ':' in str(row['title']):
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
                if st.button("▶ Play", key=f"play_{row['title']}", use_container_width=True):
                    on_play(st.session_state.user_id, row['title'])
                    st.toast(f"Playing {show_name.strip()}...")
            with btn_col2:
                label = "✅ Saved" if st.session_state[save_key] else "🔖 Save"
                if st.button(label, key=f"btn_{row['title']}", use_container_width=True):
                    st.session_state[save_key] = not st.session_state[save_key]
                    on_save(st.session_state.user_id, row['title'], st.session_state[save_key])
                    st.rerun()

        st.divider()

def on_play(user_id, title):
    # TODO: log play event to view history
    pass

def on_save(user_id, title, saved: bool):
    # TODO: update save status in view history
    pass

# --- page ---

st.title("Get personalized recommendations based on your favorite genres")
st.write("You'll receive recommendations for the genres you select based on your watch history. If you haven't watched anything yet, we'll use your initial selections to start personalizing your recommendations.")

view_history, programs, programs_tfidf = load_recommendation_data()

# user selector — stores in session state so other pages can access it
if "user_id" not in st.session_state:
    st.session_state.user_id = None

st.session_state.user_id = st.selectbox(
    "Select user",
    view_history['user_id'].unique(),
    index=list(view_history['user_id'].unique()).index(st.session_state.user_id) if st.session_state.user_id else 0
)

# lambda slider — also stored in session state for consistency with app.py
if "lambda_param" not in st.session_state:
    st.session_state.lambda_param = 0.5

alpha = st.select_slider(
    "Recommendation style",
    options=["More personalized", "Balanced", "More diverse"],
    value="Balanced"
)
alpha_map = {"More personalized": 0.2, "Balanced": 0.5, "More diverse": 0.8}
st.session_state.lambda_param = alpha_map[alpha]

# genre selection
genre_icons = {
    "arts": "🎨",
    "cbbc": "🧒",
    "comedy": "😄",
    "documentaries": "🎬",
    "entertainment": "🎭",
    "films": "🎥",
    "from-the-archives": "📼",
    "history": "🏛️",
    "lifestyle": "🌿",
    "music": "🎵",
    "science-and-nature": "🔬",
    "signed": "🤟",
    "sports": "⚽",
}

for genre in genre_icons:
    if f"genre_{genre}" not in st.session_state:
        st.session_state[f"genre_{genre}"] = False

st.markdown("### Pick your genres")

cols = st.columns(3, gap="large")
for i, (genre, icon) in enumerate(genre_icons.items()):
    key = f"genre_{genre}"
    label = genre.replace("-", " ").title()

    with cols[i % 3]:
        col_icon, col_check = st.columns([1, 4])
        with col_icon:
            st.markdown(f"<div style='font-size: 24px; padding-top: 4px;'>{icon}</div>", unsafe_allow_html=True)
        with col_check:
            st.checkbox(label, key=key)

st.divider()

selected_genres = [g for g in genre_icons if st.session_state[f"genre_{g}"]]
if selected_genres:
    st.markdown(f"**Selected:** {', '.join(g.replace('-', ' ').title() for g in selected_genres)}")
else:
    st.caption("No genres selected yet.")

# if st.button("Get recommendations", type="primary", disabled=len(selected_genres) == 0):
#     with st.spinner("Finding recommendations..."):
#         results = get_genre_recommendations(
#             st.session_state.user_id,
#             selected_genres,
#             st.session_state.lambda_param
#         )
#     if results is None or results.empty:
#         st.info("No recommendations found for the selected genres.")
#     else:
#         display_recommendations(results)


if st.button("Get recommendations", type="primary", disabled=len(selected_genres) == 0):
    with st.spinner("Finding recommendations..."):
        results = get_genre_recommendations(
            st.session_state.user_id,
            selected_genres,
            st.session_state.lambda_param
        )
    if results is None or results.empty:
        st.info("Not enough watch history for the selected genres. Try selecting more genres or watch some content first.")
    else:
        display_recommendations(results)