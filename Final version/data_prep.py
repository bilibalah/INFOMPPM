import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

programs_raw = pd.read_csv("data/bbc_seen_items.csv").drop_duplicates("title")
programs_raw.reset_index(names="program_id").to_csv("data/programs.csv", index=False)

synthetic_data = pd.read_csv("data/synthetic_view_history_full.csv")
programs = pd.read_csv("data/programs.csv")

synthetic_data['user_id'] = synthetic_data['user_id'].astype('str')
mask = (synthetic_data['user_id'].str.len() <= 4)
synthetic_data = synthetic_data.loc[mask]

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