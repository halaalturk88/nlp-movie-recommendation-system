import pandas as pd
import ast
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer


# Load datasets
movies = pd.read_csv('datasets/tmdb_5000_movies.csv')
credits = pd.read_csv('datasets/tmdb_5000_credits.csv')


# Merge datasets
movies = movies.merge(credits, on='title')


# Select important columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]


# Remove missing values
movies.dropna(inplace=True)


# Convert genres and keywords from text format to list of names
def convert(text):
    result = []

    for item in ast.literal_eval(text):
        result.append(item['name'])

    return result


# Get first 3 actors from cast
def convert_cast(text):
    result = []
    counter = 0

    for item in ast.literal_eval(text):
        if counter < 3:
            result.append(item['name'])
            counter += 1
        else:
            break

    return result


# Get director name from crew
def get_director(text):
    result = []

    for item in ast.literal_eval(text):
        if item['job'] == 'Director':
            result.append(item['name'])
            break

    return result


# Remove spaces from multi-word terms
def remove_spaces(word_list):
    result = []

    for word in word_list:
        result.append(word.replace(" ", ""))

    return result


# Stemming function
ps = PorterStemmer()

def stem_text(text):
    stemmed_words = []

    for word in text.split():
        stemmed_words.append(ps.stem(word))

    return " ".join(stemmed_words)


# Apply conversion
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(get_director)


# Remove spaces
movies['genres'] = movies['genres'].apply(remove_spaces)
movies['keywords'] = movies['keywords'].apply(remove_spaces)
movies['cast'] = movies['cast'].apply(remove_spaces)
movies['crew'] = movies['crew'].apply(remove_spaces)


# Convert overview text into list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())


# Create tags column using multiple textual features
movies['tags'] = (
    movies['overview'] +
    movies['genres'] +
    movies['keywords'] +
    movies['cast'] +
    movies['crew']
)


# Convert tags list into one string
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))


# Convert all tags to lowercase
movies['tags'] = movies['tags'].apply(lambda x: x.lower())


# Apply stemming
movies['tags'] = movies['tags'].apply(stem_text)


# Convert titles to lowercase for easier search
movies['title_lower'] = movies['title'].str.lower()


# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = tfidf.fit_transform(movies['tags'])


# Cosine Similarity
similarity = cosine_similarity(vectors)


# Helper function to find movie using exact match or fuzzy matching
def find_movie_index(movie):
    movie = movie.lower().strip()
    all_titles = movies['title_lower'].values

    # Exact match
    if movie in all_titles:
        matched_title = movie

    # Fuzzy match
    else:
        matches = difflib.get_close_matches(
            movie,
            all_titles,
            n=1,
            cutoff=0.4
        )

        if len(matches) == 0:
            return None

        matched_title = matches[0]

    index = movies[movies['title_lower'] == matched_title].index[0]

    return index


# Recommendation function with similarity score for each recommended movie
def recommend(movie):
    index = find_movie_index(movie)

    if index is None:
        return []

    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for movie_index, score in movie_list:
        recommendations.append({
            "title": movies.iloc[movie_index].title,
            "score": round(score, 3),
            "percentage": round(score * 100, 1)
        })

    return recommendations


# Evaluation function to return the matched movie title
def evaluate_recommendations(movie):
    index = find_movie_index(movie)

    if index is None:
        return {
            "matched_movie": "Movie not found"
        }

    return {
        "matched_movie": movies.iloc[index].title
    }