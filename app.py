from flask import Flask, render_template, request
from recommendation import recommend, evaluate_recommendations

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():

    recommendations = []
    evaluation = None

    if request.method == 'POST':

        movie = request.form['movie']

        recommendations = recommend(movie)
        evaluation = evaluate_recommendations(movie)

    return render_template(
        'index.html',
        recommendations=recommendations,
        evaluation=evaluation
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)