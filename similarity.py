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
    'data/science-and-nature.pkl'
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
    filtered_users = user_df[user_df['title'].isin(valid_titles)]
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
        return f"cold start required for user"