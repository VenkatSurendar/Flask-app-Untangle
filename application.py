from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from keybert import KeyBERT
import json

application = app = Flask(__name__)

model = KeyBERT(model="distilbert-base-nli-mean-tokens")

# app = Flask(__name__)


df = pd.concat(
    map(pd.read_csv, ['reviews-13495-13500.csv', 'reviews-13500-13537.csv']), ignore_index=True)


@application.route('/', methods=['GET', 'POST'])
def home():
    if(request.method == 'GET'):

        data = "hello world"
        return jsonify({'data': data})


@application.route('/home/<int:num>', methods=['GET'])
def disp(num):
    print(num)
    return jsonify({'data': num**2})


# No of reviews
@application.route('/no_of_reviews/<int:appid>')
def avg_playtime(appid):

    df3 = df[(df['appid'] == appid)]
    x = df3.shape[0]
    print(x)
    return jsonify({'no_of_reviews': x})


# True , False sentiments
@application.route('/review_sentiments/<int:appid>')
def rev_sentiments(appid):

    df3 = df[(df['appid'] == appid)]

    y = df3['voted_up'].value_counts().tolist()
    print(y)
    return jsonify({'sentiment': y})


# recent reviews


@application.route('/recent_reviews/<int:appid>')
def recent_reviews(appid):

    df3 = df[(df['appid'] == appid)]

    sorted_df = df3.sort_values('unix_timestamp_created', ascending=[False])
    z = sorted_df.values.tolist()
    return jsonify({'Recent_reviws': z})

# trending keywords


@application.route('/trending_keywords/<int:appid>')
def trending_keywords(appid):

    df3 = df[(df['appid'] == appid)]
    x = df3.shape[0]
    arr = []
    for i in range(1, 100):
        a = model.extract_keywords(
            df3['review'].values[i], top_n=1, keyphrase_ngram_range=(1, 1), stop_words="english")
        arr = arr + a
    x = arr.sort(key=lambda x: x[1])
    print(x)
    return jsonify({'Trending keywords': arr})

# Average rating of game


@application.route('/avg_rating/<int:appid>')
def avg_rating(appid):

    df3 = df[(df['appid'] == appid)]
    x = df3['voted_up'].value_counts().tolist()
    rate = (x[0] * 5 + (x[1])) / (x[0]+x[1])
    return jsonify({'Average Rating': rate})


@application.route('/top_games')
def top_games():
    y = df.appid.unique()
    z = []
    for ele in y:
        df3 = df[(df['appid'] == ele)]

        x = df3['voted_up'].value_counts().tolist()

        if len(x) == 2:
            z.append((x[0] * 5 + (x[1])) / (x[0]+x[1]))
    top = list(map(lambda x, y: (x, y), y, z))
    top_sorted = sorted(top, key=lambda x: x[1])

    print(top_sorted)
    # my_array = np.asarray(top_sorted)
    return {'Top games': "heelo"}


# driver function
if __name__ == '__main__':

    application.run(debug=True)
