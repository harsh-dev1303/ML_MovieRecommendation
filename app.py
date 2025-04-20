# app.py

from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model and data
with open("movie_model.pkl", "rb") as f:
    cosine_sim, movies, indices = pickle.load(f)


# Recommendation function
def get_recommendations(title):
    if title not in indices:
        return ["Movie not found."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5
    movie_indices = [i[0] for i in sim_scores]
    return movies["title"].iloc[movie_indices].tolist()


@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    movie_titles = movies["title"].tolist()
    if request.method == "POST":
        movie_title = request.form["movie"]
        recommendations = get_recommendations(movie_title)
    return render_template(
        "index.html", recommendations=recommendations, movies=movie_titles
    )


if __name__ == "__main__":
    app.run(debug=True)
