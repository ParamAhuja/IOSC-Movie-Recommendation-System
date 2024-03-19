
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
from surprise import Dataset, Reader, SVD
import uuid
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation App")

# Create a title
st.title('Movie Recommendation System')
st.write("Check Inception movie")


#st.sidebar.header('User Input Features')
#st.write("Check Inception movie")
# User input for movie name
movie_name = st.text_input('Enter a movie name')
movie_name = movie_name.title()
st.write(movie_name)
rating = st.slider('Rate the movie', 1, 5)



# Recommend similar movies and show the collaborative filter rating
if st.button('Recommend: '):
  left, right = st.columns(2)
  with left:
    st.header("Similar Movie Recommender:")

    df1=pd.read_csv('/content/gdrive/MyDrive/input/tmdb-movie-metadata/tmdb_5000_credits.csv')
    df2=pd.read_csv('/content/gdrive/MyDrive/input/tmdb-movie-metadata/tmdb_5000_movies.csv')
    df1.columns = ['id','tittle','cast','crew']
    df2= df2.merge(df1,on='id')


    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        df2[feature] = df2[feature].apply(literal_eval)


    # Get the director's name from the crew feature. If director is not listed, return NaN
    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    # Returns the list top 3 elements or entire list; whichever is more.
    def get_list(x):
        if isinstance(x, list):
            names = [i['name'] for i in x]
            #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
            if len(names) > 3:
                names = names[:3]
            return names

        #Return empty list in case of missing/malformed data
        return []

    # Define new director, cast, genres and keywords features that are in a suitable form.
    df2['director'] = df2['crew'].apply(get_director)

    features = ['cast', 'keywords', 'genres']
    for feature in features:
        df2[feature] = df2[feature].apply(get_list)

    # Function to convert all strings to lower case and strip names of spaces
    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            #Check if director exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''

    # Apply clean_data function to your features.
    features = ['cast', 'keywords', 'director', 'genres']

    for feature in features:
        df2[feature] = df2[feature].apply(clean_data)

    def create_soup(x):
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
    df2['soup'] = df2.apply(create_soup, axis=1)

    # Import CountVectorizer and create the count matrix
    from sklearn.feature_extraction.text import CountVectorizer

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df2['soup'])

    # Compute the Cosine Similarity matrix based on the count_matrix
    from sklearn.metrics.pairwise import cosine_similarity

    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


    # Reset index of our main DataFrame and construct reverse mapping as before
    df2 = df2.reset_index()
    indices = pd.Series(df2.index, index=df2['title'])

    # Function that takes in movie title as input and outputs most similar movies
    def get_recommendations(title, cosine_sim=cosine_sim2):
        # Get the index of the movie that matches the title
        idx = indices[title]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        return df2['title'].iloc[movie_indices]

    try:
      st.write(get_recommendations(movie_name))
      #st.write("correct2")
    except:
      st.write("Oopsie! Movie not found in the database.")


  with right:
    st.header("Collaborative Filtering Movie Ratings:")
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
    # Get movie ID from title
    #st.write("correct1")

    try:
      movie_id = movie_dict[movie_name]
      #movie_id = data.raw_data[data.raw_data['name'] == movie_name]['movieId'].values[0]
      #st.write("correct2")
      # Predict the rating
      prediction = svd.predict(1, movie_id, rating).est
      #st.write("correct3")

      # Display the prediction
      st.write('Predicted rating:', prediction)

    except:
      st.write("Oopsie! Movie not found in the database.")

