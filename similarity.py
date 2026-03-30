import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the synthetic view history dataset
df_synth_simple = pd.read_csv('synthetic_view_history_simple.csv')

# Load the documentary, comedy, sports, and science-and-nature datasets
files_to_load = [
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
    'data/signed.pkl'

]
df_list = [pd.read_pickle(file) for file in files_to_load]
df = pd.concat(df_list, ignore_index=True)
df = df.dropna(subset=['synopsis_large']).reset_index(drop=True)

#takes a dataframe and a text columns and returns similarity matrix, tf-idf matric and vectorizer
def similarity_engine(df, text_column='synopsis_large', max_features=5000):
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(df[text_column])
    item_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return item_similarity_matrix, tfidf_matrix, tfidf

#hybrid engine for content and collaborative 
def hybrid_engine(df, user_df, content_similarity_matrix, alpha=0.5):
    users = user_df.copy()
    #1 point for watching, 2 points for explicitly saving
    users['interaction_weight'] = 1
    users.loc[users['save'] == 'yes', 'interaction_weight'] = 2
    #filter to only include shows we actually loaded
    valid_titles = df['title'].tolist()
    filtered_users = users[user_df['title'].isin(valid_titles)]
    #collaboraitve matrix
    user_item_matrix = filtered_users.pivot_table(
    index='title', 
    columns='user_id', 
    values='interaction_weight', 
    fill_value=0
)
    #aligning matrix order
    user_item_matrix = user_item_matrix.reindex(df['title']).fillna(0)
    collab_similarity_matrix = cosine_similarity(user_item_matrix)

    print(f"3. Blending Hybrid Matrix (Content Weight: {alpha * 100}%)...")
    hybrid_similarity_matrix = (alpha * content_similarity_matrix) + ((1 - alpha) * collab_similarity_matrix)
    
    return collab_similarity_matrix, hybrid_similarity_matrix


def mmr_recommendations(target_user_id, user_data, df, fairness_matrix, content_matrix, top_n=5, lambda_param=0.5):
    user_history = user_data[user_data['user_id'] == target_user_id]
    saved_shows = user_history[user_history['save'] == 'yes']['title'].tolist()

    if not saved_shows:
        return "Cold start required for user"

    # get indices of shows the user has already seen
    seen_titles = user_history['title'].tolist()
    seen_indices = df[df['title'].isin(seen_titles)].index.tolist()

    # get indices of saved shows to use as the query
    saved_indices = df[df['title'].isin(saved_shows)].index.tolist()

    if not saved_indices:
        return "No saved shows found in catalogue"

    # candidate pool: everything the user hasn't seen
    candidate_indices = [i for i in df.index if i not in seen_indices]

    if not candidate_indices:
        return "No unseen candidates available"

    # relevance score: average similarity to saved shows
    relevance_scores = fairness_matrix[candidate_indices][:, saved_indices].mean(axis=1)

    # MMR loop: iteratively pick the best candidate balancing relevance and diversity
    selected = []
    candidate_indices = list(candidate_indices)

    while len(selected) < top_n and candidate_indices:
        if not selected:
            # first pick: highest relevance
            best_idx = candidate_indices[int(np.argmax(relevance_scores))]
        else:
            # MMR score: balance relevance vs similarity to already selected
            mmr_scores = []
            for i, idx in enumerate(candidate_indices):
                relevance = relevance_scores[i]
                # max similarity to already selected items
                similarity_to_selected = max(
                    content_matrix[idx][s] for s in selected
                )
                mmr = lambda_param * relevance - (1 - lambda_param) * similarity_to_selected
                mmr_scores.append(mmr)
            best_idx = candidate_indices[int(np.argmax(mmr_scores))]

        selected.append(best_idx)
        pos = candidate_indices.index(best_idx)
        candidate_indices.pop(pos)
        relevance_scores = np.delete(relevance_scores, pos)

    return df.iloc[selected][['title', 'category', 'description', 'synopsis_small', 'synopsis_large', 'image', 'duration_txt']].reset_index(drop=True)