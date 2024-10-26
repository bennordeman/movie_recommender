from flask import Blueprint, request, render_template, jsonify
from app.recommender import get_recommendations_based_on_movie

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/recommend', methods=['POST'])
def recommend():
    liked_movie = request.form.get('movie_title')
    recommendations = get_recommendations_based_on_movie(liked_movie)
    
    return render_template('recommendations.html', liked_movie=liked_movie, recommendations=recommendations)
