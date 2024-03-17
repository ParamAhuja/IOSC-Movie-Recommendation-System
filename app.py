
import streamlit as st
import pandas as pd
import pickle
import requests
from surprise import Dataset, Reader, SVD

#reader = Reader(rating_scale=(1, 5))
#ratings_df = pd.read_csv('/content/gdrive/MyDrive/input/the-movies-dataset/ratings.csv')
#data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)


# Load the model from pickle
svd = pickle.load(open('surprise_model.pkl', 'rb'))

# Get the movie data
movies = pd.read_csv('/content/gdrive/MyDrive/input/the-movies-dataset/movies_metadata.csv')

# Create a dictionary to map movie titles to IDs
movie_dict = {movie[20]: movie[5] for movie in movies.values}
# st.write((x, movie_dict[x]) for x in movie_dict.keys())

# Create a title and sidebar
st.title('Movie Recommendation System')
st.sidebar.header('User Input Features')

# User input for movie name
movie_name = st.text_input('Enter a movie name')
rating = st.sidebar.slider('Rate the movie', 1, 5)

# Recommend movies
if st.button('Recommend'):
    # Get movie ID from title
    st.write("correct1")
    
    movie_id = movie_dict[movie_name]
    #movie_id = data.raw_data[data.raw_data['name'] == movie_name]['movieId'].values[0]
    
    st.write("correct2")

    # Predict the rating
    prediction = svd.predict(1, movie_id, rating).est
    st.write("correct3")

    # Display the prediction
    st.write('Predicted rating:', prediction)
