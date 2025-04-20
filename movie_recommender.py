# movie_recommender.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

# Step 1: Load the datasets
movies = pd.read_csv('movies.csv')        # movieId, title, genres
tags = pd.read_csv('tags.csv')            # userId, movieId, tag

# Step 2: Combine tags with movies
tags['tag'] = tags['tag'].astype(str)
tags_agg = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
movies = movies.merge(tags_agg, on='movieId', how='left')
movies['tag'] = movies['tag'].fillna('')

# Step 3: Combine genres and tags into a new column
movies['combined'] = movies['genres'] + ' ' + movies['tag']

# Step 4: TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])

# Step 5: Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 6: Create a movie index mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Step 7: Recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return ["Movie not found in dataset."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 recommendations (excluding itself)
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Optional: Test it locally
if __name__ == "__main__":
    movie_name = input("Enter a movie title: ")
    recommendations = get_recommendations(movie_name)
    print("\nRecommended movies:")
    for rec in recommendations:
        print(rec)

# Step 8: Save the model and data needed
with open('movie_model.pkl', 'wb') as f:
    pickle.dump((cosine_sim, movies, indices), f)
