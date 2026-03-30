import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.chdir("/home/anass/university/msc_applied_data_science/INFOMPPM/INFOMPPM")

synthetic_data = pd.read_csv("data/raw_data/synthetic_view_history_full.csv")
synthetic_data[["user_id", "persona"]].drop_duplicates("user_id").to_csv("data/users.csv", index=False)

programs_raw = pd.read_csv("data/raw_data/bbc_seen_items.csv").drop_duplicates("title")
programs_raw.reset_index(names="program_id").to_csv("data/programs.csv", index=False)


synthetic_data = pd.read_csv("data/raw_data/synthetic_view_history_full.csv")
programs = pd.read_csv("data/programs.csv")

temp = synthetic_data
synthetic_data = pd.merge(temp, programs[["program_id", "title"]], on="title", how="left")

synthetic_data[["user_id", "program_id", "listen_ratio", "save"]].to_csv("data/view_history.csv", index=False)


def similarity_engine(df, text_column='synopsis_large', max_features=500):
    # takes a dataframe and a text columns and returns similarity matrix, tf-idf matrix and vectorizer
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(df[text_column])
    return tfidf_matrix, tfidf

programs = pd.read_csv("data/programs.csv")
tfidf_matrix, tfidf = similarity_engine(programs)

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
tfidf_df.reset_index(names="program_id").to_csv("data/programs_tfidf.csv", index=False)



view_history = pd.read_csv("data/view_history.csv")
programs_tfidf = pd.read_csv("data/programs_tfidf.csv")

user_profiles = pd.merge(view_history, programs_tfidf, on="program_id", how="left")
user_profiles.dropna(axis=0)

tfidf_cols = user_profiles.columns[user_profiles.columns.get_loc('save') + 1:]
user_profiles[tfidf_cols] = user_profiles[tfidf_cols].multiply(user_profiles['listen_ratio'], axis=0)

user_profiles = user_profiles.groupby('user_id')[tfidf_cols].mean().reset_index()
user_profiles.to_csv("data/user_profiles.csv", index=False)

user_profiles = pd.read_csv("data/user_profiles.csv")
programs = pd.read_csv("data/programs_tfidf.csv")

tfidf_cols = [col for col in user_profiles.columns if col != 'user_id']
user_vectors = user_profiles[tfidf_cols].values
program_vectors = programs[tfidf_cols].values

sim_matrix = cosine_similarity(user_vectors, program_vectors)

sim_df = pd.DataFrame(
    sim_matrix,
    index=user_profiles['user_id'],
    columns=programs['program_id']
)

sim_df.to_csv("data/content_relevance_score.csv", index=False)

user_scores = sim_df.loc['U100']
user_scores.sort_values(ascending=False).head(20)

# Adjust this threshold to whatever we need for the exposure fairness and diversity/mmr
threshold_amount = 500
top_recommendations = sim_df.apply(lambda row: row.rank(ascending=False) <= threshold_amount, axis=1).astype(int)
top_recommendations.to_csv("top_recommendation_content.csv", index=False)
