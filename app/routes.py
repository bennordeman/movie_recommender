from flask import Blueprint, request, render_template, jsonify
from app.recommender import get_recommendations_based_on_movie

main = Blueprint('main', __name__)

@main.route('/')
def home():
    # Render the home page with a form
    return render_template('index.html')

@main.route('/recommend', methods=['POST'])
def recommend():
    # Get the movie title entered by the user
    liked_movie = request.form.get('movie_title')
    recommendations = get_recommendations_based_on_movie(liked_movie)
    
    # Render the result on a new page
    return render_template('recommendations.html', liked_movie=liked_movie, recommendations=recommendations)
