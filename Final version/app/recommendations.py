import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def user_profile_def(user_id, view_history, programs, programs_tfidf, genres):
    view_history_user = view_history[view_history["user_id"] == user_id]

    view_history_user = pd.merge(
        view_history_user,
        programs[["program_id", "category"]],
        on="program_id",
        how="left"
    )

    # if no watch history in selected genres, fall back to all watched genres
    # but keep the selected genres for candidate filtering
    watched_in_genres = view_history_user[view_history_user['category'].isin(genres)]
    if len(watched_in_genres) == 0:
        profile_genres = view_history_user["category"].dropna().unique().tolist()
    else:
        profile_genres = genres  # use selected genres for profile building

    view_history_user = view_history_user.loc[view_history_user['category'].isin(profile_genres)]
    view_history_user = view_history_user.drop('category', axis=1)

    user_profile = pd.merge(view_history_user, programs_tfidf, on="program_id", how="left")
    user_profile = user_profile.dropna(axis=0)
    user_profile.loc[user_profile['save'] == 'yes', 'listen_ratio'] *= 2
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


def top_50_recommendations(user_id, similarity_df, view_history, genres, programs, threshold_value=0.6):
    similarity_df = similarity_df.transpose()
    similarity_df.index.name = 'program_id'
    similarity_df = similarity_df.reset_index()
    similarity_df = similarity_df.sort_values(by=user_id, ascending=False)

    similarity_df = pd.merge(similarity_df, programs, on="program_id", how="left").dropna()
    similarity_df = similarity_df[~similarity_df['program_id'].isin(
        view_history[view_history["user_id"] == user_id]['program_id']
    )]
    similarity_df = similarity_df.loc[similarity_df['category'].isin(genres)]

    return similarity_df[similarity_df[user_id] >= threshold_value]


def exposure_fairness(top_50_recommendations):
    top_50_recommendations = top_50_recommendations.sort_values(
        by="inclusion_score", ascending=False
    ).reset_index(drop=True)
    return top_50_recommendations[:50]


def content_diversity(fairness_df, programs_tfidf, top_n=15, lambda_param=0.5):
    candidate_ids = fairness_df['program_id'].tolist()

    if not candidate_ids:
        return pd.DataFrame()

    candidate_tfidf = programs_tfidf[programs_tfidf['program_id'].isin(candidate_ids)].set_index('program_id')
    candidate_tfidf = candidate_tfidf.reindex(candidate_ids)

    tfidf_cols = [col for col in candidate_tfidf.columns if col != 'user_id']
    item_vectors = candidate_tfidf.loc[candidate_ids, tfidf_cols].values

    sim_matrix = cosine_similarity(item_vectors)
    content_matrix = pd.DataFrame(sim_matrix, index=candidate_ids, columns=candidate_ids)

    selected_ids = []

    first_pick_id = candidate_ids[0]
    selected_ids.append(first_pick_id)
    candidate_ids.remove(first_pick_id)

    max_score = fairness_df['inclusion_score'].max()
    max_score = max_score if max_score > 0 else 1

    while len(selected_ids) < top_n and candidate_ids:
        best_mmr = -float('inf')
        best_candidate_id = None

        for candidate_id in candidate_ids:
            raw_relevance = fairness_df.loc[
                fairness_df['program_id'] == candidate_id, 'inclusion_score'
            ].values[0]
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


def build_collab_matrix(view_history, programs):
    """builds a program x program collaborative similarity matrix from view history"""
    users = view_history.copy()
    users['interaction_weight'] = 1
    users.loc[users['save'] == 'yes', 'interaction_weight'] = 2

    valid_ids = programs['program_id'].tolist()
    users = users[users['program_id'].isin(valid_ids)]

    user_item_matrix = users.pivot_table(
        index='program_id',
        columns='user_id',
        values='interaction_weight',
        fill_value=0
    )
    user_item_matrix = user_item_matrix.reindex(valid_ids).fillna(0)
    collab_matrix = cosine_similarity(user_item_matrix)
    collab_matrix = pd.DataFrame(collab_matrix, index=valid_ids, columns=valid_ids)

    return collab_matrix


def build_content_matrix(programs, programs_tfidf):
    """builds a program x program content similarity matrix from tfidf"""
    tfidf_cols = [col for col in programs_tfidf.columns if col != 'program_id']
    valid_ids = programs['program_id'].tolist()

    tfidf_aligned = programs_tfidf[programs_tfidf['program_id'].isin(valid_ids)].set_index('program_id')
    tfidf_aligned = tfidf_aligned.reindex(valid_ids).fillna(0)

    item_vectors = tfidf_aligned[tfidf_cols].values
    sim_matrix = cosine_similarity(item_vectors)
    content_matrix = pd.DataFrame(sim_matrix, index=valid_ids, columns=valid_ids)

    return content_matrix


def mmr_collaborative(user_id, view_history, programs, active_matrix, content_matrix, top_n=15, lambda_param=0.5):
    """MMR recommendations using collaborative or hybrid matrix for relevance, content matrix for diversity penalty"""
    user_history = view_history[view_history['user_id'] == user_id]
    saved_ids = user_history[user_history['save'] == 'yes']['program_id'].tolist()
    seen_ids = user_history['program_id'].tolist()

    if not saved_ids:
        return pd.DataFrame()

    valid_ids = programs['program_id'].tolist()
    saved_ids = [i for i in saved_ids if i in valid_ids]
    candidate_ids = [i for i in valid_ids if i not in seen_ids]

    if not candidate_ids or not saved_ids:
        return pd.DataFrame()

    # relevance: average similarity to saved programs
    relevance_scores = active_matrix.loc[candidate_ids, saved_ids].mean(axis=1).values

    selected_ids = []
    candidate_ids = list(candidate_ids)

    while len(selected_ids) < top_n and candidate_ids:
        if not selected_ids:
            best_idx = candidate_ids[int(np.argmax(relevance_scores))]
        else:
            mmr_scores = []
            for i, cid in enumerate(candidate_ids):
                relevance = relevance_scores[i]
                similarity_to_selected = content_matrix.loc[cid, selected_ids].max()
                mmr = lambda_param * relevance - (1 - lambda_param) * similarity_to_selected
                mmr_scores.append(mmr)
            best_idx = candidate_ids[int(np.argmax(mmr_scores))]

        selected_ids.append(best_idx)
        pos = candidate_ids.index(best_idx)
        candidate_ids.pop(pos)
        relevance_scores = np.delete(relevance_scores, pos)

    final_recs = programs[programs['program_id'].isin(selected_ids)].copy()
    final_recs = final_recs.set_index('program_id').loc[selected_ids].reset_index()

    return final_recs[['program_id', 'title', 'category', 'synopsis_small', 'image', 'duration_txt']]


def recommendation_content(user_id, view_history, programs, programs_tfidf, genres, lambda_param):
    if len(genres) == 0:
        genres = pd.merge(view_history, programs, on="program_id", how="left")["category"].unique().tolist()

    user_profile = user_profile_def(user_id, view_history, programs, programs_tfidf, genres)
    similarity_df = cosine_similarity_def(user_profile, programs_tfidf)
    top50_programs = top_50_recommendations(user_id, similarity_df, view_history, genres, programs)
    exposure = exposure_fairness(top50_programs)
    final_recs = content_diversity(exposure, programs_tfidf, top_n=15, lambda_param=lambda_param)

    if final_recs.index.name == 'program_id':
        final_recs = final_recs.reset_index()

    # columns already present from top_50 merge, just clean up
    final_recs = final_recs.rename(columns={
        'title_x': 'title',
        'category_x': 'category',
        'synopsis_small_x': 'synopsis_small',
        'image_x': 'image',
        'duration_txt_x': 'duration_txt'
    })

    # drop the duplicate _y columns
    final_recs = final_recs[[col for col in final_recs.columns if not col.endswith('_y')]]

    return final_recs


def recommendation_collaborative(user_id, view_history, programs, programs_tfidf, genres, lambda_param, alpha=0.5):
    """
    alpha controls the blend: 0.0 = pure collaborative, 1.0 = pure content, 0.5 = hybrid
    lambda_param controls MMR diversity: 0.0 = max diversity, 1.0 = max relevance
    """
    if len(genres) > 0:
        valid_program_ids = programs[programs['category'].isin(genres)]['program_id'].tolist()
        programs_filtered = programs[programs['program_id'].isin(valid_program_ids)]
    else:
        programs_filtered = programs

    collab_matrix = build_collab_matrix(view_history, programs_filtered)
    content_matrix = build_content_matrix(programs_filtered, programs_tfidf)

    if alpha == 0.0:
        active_matrix = collab_matrix
    elif alpha == 1.0:
        active_matrix = content_matrix
    else:
        active_matrix = (alpha * content_matrix) + ((1 - alpha) * collab_matrix)

    return mmr_collaborative(user_id, view_history, programs_filtered, active_matrix, content_matrix, top_n=15, lambda_param=lambda_param)


if __name__ == "__main__":
    view_history = pd.read_csv("../data/view_history.csv")
    programs = pd.read_csv("../data/programs.csv")
    programs_tfidf = pd.read_csv("../data/programs_tfidf.csv")

    print("Testing content-based:")
    print(recommendation_content("U99", view_history, programs, programs_tfidf, [], 0.5))

    print("Testing collaborative:")
    print(recommendation_collaborative("U99", view_history, programs, programs_tfidf, [], 0.5))