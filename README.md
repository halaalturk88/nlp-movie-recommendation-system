# Movie Recommendation System Using NLP

**Project type:** University NLP coursework project  
**Domain:** Natural Language Processing · Content-Based Recommendation · Web Application  
**Stack:** Python, Pandas, scikit-learn, NLTK, Flask, HTML, CSS

## Overview

This project recommends **five movies with similar textual content** to a user-selected movie. It uses movie metadata rather than watch histories or personal ratings. A small Flask interface lets users search by movie title and view ranked recommendations.

## How it works

1. Read `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` and merge them by title.
2. Extract overview, genres, keywords, the first three cast members, and director.
3. Clean text, combine metadata into `tags`, lowercase, and apply Porter stemming.
4. Convert tags into TF-IDF features (up to 5,000 terms, with English stop words removed).
5. Compute cosine similarity between movies and return the five highest-scoring **other** movies.
6. Match approximate input titles using Python's `difflib` and display recommendations in Flask.

### Interpretation of the displayed percentage

The percentage displayed in the interface is **cosine similarity × 100**. It does **not** represent recommendation accuracy, user satisfaction, or the probability that a viewer will like the movie. No external relevance benchmark was provided in this repository.

## Project files

```text
app.py                         # Flask routes
recommendation.py              # NLP preprocessing and recommendation pipeline
static/style.css               # Interface styling
templates/index.html           # HTML template
datasets/README.md             # Dataset instructions (CSV files are not included)
docs/NLP_Movie_Recommendation_Project_Explanation.docx
requirements.txt
.gitignore
```

## Run locally

Python 3 is required. From the root of the project:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Obtain the **TMDB 5000 Movie Dataset** from a source where you are permitted to use it, and place these CSVs under `datasets/`:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Then run:

```bash
python app.py
```

Open `http://localhost:10000`. Run it from the repository's root directory because the existing code uses relative dataset paths.

## Features

- Content-based recommendations (top 5)
- TF-IDF and cosine similarity
- Text and metadata preprocessing
- Fuzzy title matching for approximate searches
- Simple responsive Flask interface

## Limitations and potential improvements

- Content similarity is not personalized, and there is no labeled relevance evaluation.
- The similarity matrix is built when the Python module loads, so startup time and memory usage can grow as the dataset grows.
- The existing fuzzy match can select an unintended title for ambiguous queries.
- Potential next steps: a stronger evaluation set, sentence embeddings, caching or nearest-neighbor search, and better error handling for missing datasets.

## Academic context

Created as an NLP university course project. Credit and individual contributions should reflect the actual team arrangement.

**Verification note:** The original code was reviewed for structure and syntax, but the packaged project was not end-to-end tested with the two TMDB datasets.
