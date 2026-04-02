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
    similarity_df = similarity_df.transpose().sort_values(by=user_id, ascending=False)
    similarity_df = pd.merge(similarity_df, pd.read_csv("data/programs.csv"), on="program_id", how="left").dropna()
    return similarity_df.iloc[:threshold_amount]

def exposure_fairness(top_50_recommendations):
    return top_50_recommendations.sort_values(by="inclusion_score", ascending=False).reset_index(drop=True)

def content_diversity(fairness_df, programs_tfidf, top_n=15, lambda_param=0.5):
    candidate_ids = fairness_df['program_id'].tolist()
    if not candidate_ids:
        return pd.DataFrame()
    # 1. content similarity matrix for just these candidates
    candidate_tfidf = programs_tfidf[programs_tfidf['program_id'].isin(candidate_ids)].set_index('program_id')
    candidate_tfidf = candidate_tfidf.reindex(candidate_ids)
    # Exclude user_id to get pure vectors
    tfidf_cols = [col for col in candidate_tfidf.columns if col != 'user_id']
    item_vectors = candidate_tfidf.loc[candidate_ids, tfidf_cols].values
    sim_matrix = cosine_similarity(item_vectors)
    content_matrix = pd.DataFrame(sim_matrix, index=candidate_ids, columns=candidate_ids)
    selected_ids = []
    #highest fairness score
    first_pick_id = candidate_ids[0]
    selected_ids.append(first_pick_id)
    candidate_ids.remove(first_pick_id)
    # Normalize inclusion_score
    max_score = fairness_df['inclusion_score'].max()
    max_score = max_score if max_score > 0 else 1
    # 3. mmr
    while len(selected_ids) < top_n and candidate_ids:
        best_mmr = -float('inf')
        best_candidate_id = None  
        for candidate_id in candidate_ids:
            raw_relevance = fairness_df.loc[fairness_df['program_id'] == candidate_id, 'inclusion_score'].values[0]
            relevance = raw_relevance / max_score 
            max_similarity = content_matrix.loc[candidate_id, selected_ids].max()
            mmr_score = (lambda_param * relevance) - ((1 - lambda_param) * max_similarity)
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_candidate_id = candidate_id   
        selected_ids.append(best_candidate_id)
        candidate_ids.remove(best_candidate_id)
    final_recs = fairness_df.set_index('program_id').loc[selected_ids].reset_index()
    final_recs['lambda_setting'] = lambda_param
    
    return final_recs

def recommendation_content(user_id, view_history, programs_tfidf):
    user_profile = user_profile_def(user_id, view_history, programs_tfidf)
    
    similarity_df = cosine_similarity_def(user_profile, programs_tfidf)

    top50_programs = top_50_recommendations(user_id, similarity_df)

    exposure = exposure_fairness(top50_programs)

    final_recs = content_diversity(exposure, programs_tfidf, top_n=15, lambda_param=0.5)

    return final_recs

os.chdir("/home/anass/university/msc_applied_data_science/INFOMPPM/INFOMPPM")

recommendation_content("U9999", pd.read_csv("data/view_history.csv"), pd.read_csv("data/programs_tfidf.csv"))
