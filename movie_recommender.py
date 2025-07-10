# movie_recommender.py
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load vectorizer, movies, and movie vectors
vectorizer = joblib.load("vectorizer.pkl")
movies = joblib.load("movies.pkl")
movie_vectors = joblib.load("movie_vectors.pkl")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, user_text: str = Form(...)):
    user_text = user_text.strip()
    if not user_text:
        recommendations = ["Input text cannot be empty."]
    else:
        # Transform user text to vector
        user_vector = vectorizer.transform([user_text])
        
        # Compute cosine similarity with all movie vectors
        similarities = cosine_similarity(user_vector, movie_vectors).flatten()
        
        # Get indices of top 3 movies
        top_indices = similarities.argsort()[-3:][::-1]
        
        # Get movie titles
        recommendations = [movies[i]["title"] for i in top_indices]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_text": user_text,
        "recommendations": recommendations
    })

