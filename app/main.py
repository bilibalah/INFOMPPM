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

def top_50_recommendations(user_id, similarity_df, threshold_amount=50):
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

    return exposure

os.chdir("/home/anass/university/msc_applied_data_science/INFOMPPM/INFOMPPM")

recommendation_content("U9999", pd.read_csv("data/view_history.csv"), pd.read_csv("data/programs_tfidf.csv"))
