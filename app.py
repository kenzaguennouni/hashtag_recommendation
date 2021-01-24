from flask import Flask, render_template, request, flash
from hashtag_recommendation import similar_tweets ,embedding_dict, word2vec

app = Flask(__name__)
app.secret_key = "recommendation"


@app.route('/')
def index():
    return render_template('acceuil.html')


@app.route('/recommendationPage/')
def recommendationPage():
    return render_template('application.html')


@app.route('/getHashtag/', methods=['POST', 'GET'])
def getHashtag():
    comment = request.form['comment']
    print(comment)
    tweet = comment
    my_prediction = similar_tweets(tweet, embedding_dict, word2vec)
    print(my_prediction)
    return render_template('application.html', prediction=my_prediction)


if __name__ == "__main__":
    app.run()