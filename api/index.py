import json
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
import os
import keras
import numpy as np
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from sentiment_model.preprocessing.data_loader import DataLoader
import zipfile


with zipfile.ZipFile('api/sentiment_model/saved_models/sentiment.zip', 'r') as zip_ref:
    zip_ref.extractall('api/sentiment_model/saved_models')

app = Flask(__name__)
CORS(app)

model = keras.models.load_model('api/sentiment_model/saved_models/sentiment.keras')


@app.route("/")
def hello_world():
    return "Hello World!"


@app.route("/sentiment_score/<string:ticker>", methods=['POST', 'GET'])
def get_image(ticker: str):
    desired_path = os.path.join('company_data', datetime.today().strftime('%Y-%m-%d') + ticker + '.json')

    if (os.path.isfile(desired_path)):
        data = pd.read_json(desired_path)
        sentiment_score = data['sentiment score']
        articles = data['articles']
        return jsonify({
            'sentiment score': sentiment_score[0],
            'articles': articles.to_numpy().tolist()
        })
    else:
        newsapi = NewsApiClient(api_key=os.environ['NEWS_API_KEY'])
        company_name = ticker

        # /v2/top-headlines
        top_headlines = newsapi.get_everything(qintitle=company_name,
                                               from_param=(datetime.today() - timedelta(2)).strftime('%Y-%m-%d'),
                                               sort_by='popularity',
                                               to=datetime.today().strftime('%Y-%m-%d'),
                                               language='en')
        articles = top_headlines['articles'][:10]
        descriptions = np.array(
            list(map(lambda article: article['description'], articles))[:10]
        )
        fake_scores = np.zeros(len(descriptions))
        frame = np.concatenate([descriptions.reshape((-1, 1)),
                                fake_scores.reshape((-1, 1))], axis=1)

        loader = DataLoader([
            (pd.DataFrame(frame, columns=['review', 'sentiment']), None)
        ], 300)
        reviews = loader.get_encoded_dataset()

        predictions = model.predict(reviews[0])
        sentiment_score = float(np.mean(predictions))
        article_urls = list(map(lambda article: article['url'], articles))

        final_dict = {
            'sentiment score': sentiment_score,
            'articles': article_urls
        }

        with open(desired_path, 'w') as f:
            json.dump(final_dict, f)

        return jsonify(final_dict)
