# 🍿 Movie Recommender App

A simple NLP-powered web app where users type what kind of movie they feel like watching (e.g., "space and robots"),  
and get smart recommendations from a small movie database.

Built with **Python**, **FastAPI**, **scikit-learn** and a clean **HTML/CSS frontend**.

---

## ✨ Features

- Type what you feel like watching in natural language
- Find top matching movies based on text similarity
- Simple modern UI styled to feel like Netflix
- FastAPI backend, Jinja2 templates frontend
- NLP using TF-IDF vectorization and cosine similarity

---

## 🚀 How to run locally

Clone this repository and open it in VS Code:

```bash
git clone https://github.com/isaq128/Moive_recommender.git
cd Moive_recommender
code .

Then install dependencies (inside your virtual environment):
pip install -r requirements.txt

Train the vectorizer and save data:
python train_movie_recommender.py

