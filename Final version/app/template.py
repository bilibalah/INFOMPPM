import streamlit as st
import pandas as pd

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
                if st.button("▶ Play", key=f"play_{row['title']}", use_container_width=True):
                    on_play(user_id, row['title'])  # TODO: implement
                    st.toast(f"Playing {show_name.strip()}...")
            with btn_col2:
                label = "✅ Saved" if st.session_state[save_key] else "🔖 Save"
                if st.button(label, key=f"btn_{row['title']}", use_container_width=True):
                    st.session_state[save_key] = not st.session_state[save_key]
                    on_save(user_id, row['title'], st.session_state[save_key])  # TODO: implement
                    st.rerun()

        st.divider()