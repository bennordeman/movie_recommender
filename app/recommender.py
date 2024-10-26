import torch
import torch.nn as nn
import torch.optim as optim
from app.database import movies_df, get_movie_features, get_user_ratings

# Define a neural network model for content-based recommendations
class ContentBasedRecommender(nn.Module):
    def __init__(self, num_features, embedding_dim=50):
        super(ContentBasedRecommender, self).__init__()
        self.user_preference = nn.Linear(num_features, embedding_dim)
        self.movie_embedding = nn.Linear(num_features, embedding_dim)
    
    def forward(self, user_features, movie_features):
        user_vector = self.user_preference(user_features)
        movie_vector = self.movie_embedding(movie_features)
        return (user_vector * movie_vector).sum(1)

# Initialize model
all_genres = sorted(set(g for genres in movies_df['genres'] for g in genres))
num_features = len(all_genres)
embedding_dim = 50
model = ContentBasedRecommender(num_features, embedding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Function to train the model with user ratings
def train_model():
    model.train()
    user_ratings = get_user_ratings()
    movie_features = get_movie_features()

    for user_id, movie_id, rating in user_ratings:
        user_features = torch.FloatTensor([movie_features[movie_id - 1]])  # Adjust for movie indexing
        rating = torch.FloatTensor([rating])
        
        prediction = model(user_features, movie_features[movie_id - 1].view(1, -1))
        loss = criterion(prediction, rating)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Function to recommend movies based on a liked movie title
def get_recommendations_based_on_movie(movie_title, top_k=10):
    # Find the index of the liked movie
    try:
        liked_movie_idx = movies_df[movies_df['title'].str.contains(movie_title, case=False, na=False)].index[0]
    except IndexError:
        return ["Sorry, we couldn't find that movie. Try another title."]
    
    # Get the feature vector for the liked movie
    movie_features = get_movie_features()
    liked_movie_features = movie_features[liked_movie_idx].view(1, -1)
    
    # Calculate similarity with all other movies
    scores = torch.mm(liked_movie_features, movie_features.T).squeeze()
    scores = scores.numpy()

    # Sort and get top recommendations (excluding the liked movie itself)
    similar_movie_indices = scores.argsort()[::-1][1:top_k + 1]
    recommendations = [movies_df.iloc[idx]['title'] for idx in similar_movie_indices]
    
    return recommendations

# Function to recommend movies based on user preferences (if needed)
def get_recommendations(user_preferences, top_k=10):
    model.eval()
    recommendations = []
    
    with torch.no_grad():
        user_features = torch.FloatTensor(user_preferences).view(1, -1)
        movie_features = get_movie_features()
        
        scores = [model(user_features, movie).item() for movie in movie_features]
        top_movies = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        recommendations = [movies_df.iloc[idx]['title'] for idx, _ in top_movies]
    
    return recommendations
