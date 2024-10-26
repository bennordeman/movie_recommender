import pandas as pd
import torch

# Load CSV data
movies_df = pd.read_csv('Data/movies.csv')
ratings_df = pd.read_csv('Data/ratings.csv')

# Process genres into a one-hot encoding
movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|'))
all_genres = sorted(set(genre for genres in movies_df['genres'] for genre in genres))
for genre in all_genres:
    movies_df[genre] = movies_df['genres'].apply(lambda x: int(genre in x))

def get_movie_features():
    genre_columns = all_genres
    movie_features = torch.tensor(movies_df[genre_columns].values, dtype=torch.float32)
    return movie_features

def get_user_ratings():
    return ratings_df[['userId', 'movieId', 'rating']].values
