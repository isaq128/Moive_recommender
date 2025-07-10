# train_movie_recommender.py
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Tiny dataset: movie titles + plot summaries
movies = [
    {"title": "Interstellar", "description": "A team travels through a wormhole in space to save humanity."},
    {"title": "The Notebook", "description": "A romantic drama about a couple's lifelong love story."},
    {"title": "The Avengers", "description": "Superheroes team up to save the world from an alien invasion."},
    {"title": "Inception", "description": "A thief enters people's dreams to steal secrets and plant ideas."},
    {"title": "Titanic", "description": "A love story unfolds aboard the ill-fated Titanic ship."},
    {"title": "The Martian", "description": "An astronaut becomes stranded on Mars and must survive alone."},
    {"title": "Joker", "description": "The origin story of Batman's iconic villain, Joker."},
    {"title": "La La Land", "description": "A jazz musician and an actress fall in love in Los Angeles."},
    {"title": "Transformers", "description": "Robots from outer space battle on Earth for control."},
    {"title": "Star Wars", "description": "A space opera about Jedi knights and the battle against the dark side."},
    {"title": "Conjuring", "description": "A Horror movie where they capture spirits and talk to them"},
    {"title": "Sinister", "description": " A Horror movie about spirits killing the family"}

]

# Extract descriptions
descriptions = [movie["description"] for movie in movies]



# Vectorize
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(descriptions)

# Save vectorizer & data
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(movies, "movies.pkl")
joblib.dump(vectors, "movie_vectors.pkl")

print("Vectorizer, movie data & vectors saved!")
