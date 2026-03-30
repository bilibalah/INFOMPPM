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