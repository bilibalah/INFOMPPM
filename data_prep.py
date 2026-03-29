import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer


# Create a new csv file with unique users from the synthetic dataset

os.chdir("/home/anass/university/msc_applied_data_science/INFOMPPM/assignment")

df = pd.read_csv("data/synthetic_view_history_full.csv")
users = df[["user_id", "persona"]]
users = users.drop_duplicates()
users = users.reset_index()
users = users[["user_id", "persona"]]
df.to_csv("data/users.csv")

# Create a new csv file with view history

df = pd.read_csv("data/synthetic_view_history_full.csv")
df.to_csv("data/watch_history.csv")

# Create a new csv file with all podcasts
files_to_load = [
    'data/arts.pkl',
    'data/cbbc.pkl',
    'data/comedy.pkl',
    'data/documentaries.pkl',
    'data/entertainment.pkl',
    'data/films.pkl',
    'data/from-the-archives.pkl',
    'data/history.pkl',
    'data/lifestyle.pkl',
    'data/music.pkl',
    'data/science-and-nature.pkl',
    'data/signed.pkl',
    'data/sports.pkl'
]

df_list = [pd.read_pickle(file) for file in files_to_load]
df = pd.concat(df_list, ignore_index=True)
df = df.dropna(subset=['synopsis_large']).reset_index(names="program_id")
df.to_csv("data/programs.csv")

# Create new file of programs with tf_idf scores.

def similarity_engine(df, text_column='synopsis_large', max_features=5):
    # takes a dataframe and a text columns and returns similarity matrix, tf-idf matrix and vectorizer
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(df[text_column])
    return tfidf_matrix, tfidf

programs_df = pd.read_csv("data/programs.csv")
tfidf_matrix, tfidf = similarity_engine(programs_df)

tfidf_matrix.toarray()

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf.get_feature_names_out()
)

tfidf_df.reset_index(names="program_id").to_csv("data/programs_tfidf.csv")