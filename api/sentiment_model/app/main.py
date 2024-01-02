from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from api.sentiment_model.preprocessing.data_loader import DataLoader
from api.sentiment_model.model.fin_sentiment_analysis_model import FinSentimentModel
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

if __name__ == '__main__':
    data_1 = pd.read_csv(
        '../data/data_1_utf_8.csv'
    ).rename(
        columns={'Sentence': 'review', 'Sentiment': 'sentiment'}
    )
    data_1 = data_1[data_1['sentiment'] != 'neutral']

    data_2 = pd.read_csv(
        '../data/data_2_utf_8.csv', header=None
    ).rename(
        columns={0: 'review', 1: 'sentiment'}
    )

    data_3 = pd.read_csv(
        '../data/data_3.txt', sep='.@', header=None
    ).rename(
        columns={0: 'review', 1: 'sentiment'}
    )
    data_3 = data_3[data_3['sentiment'] != 'neutral']

    data_4 = pd.read_csv(
        '../data/data_4_utf_8.csv', header=None
    ).iloc[:, [2, 3]].rename(
        columns={2: 'sentiment', 3: 'review'}
    )
    data_4 = data_4[data_4['sentiment'] != 'Irrelevant']

    data_5 = pd.read_csv(
        '../data/data_5_utf_8.csv'
    ).iloc[:, [1, 2]].rename(
        columns={'text': 'review'}
    )

    data_5 = data_5[data_5['sentiment'] != 'neutral']

    sst_phrases = pd.read_csv(
        '../data/dictionary.txt',
        sep='|'
    )
    sst_labels = pd.read_csv(
        '../data/sentiment_labels.txt',
        sep='|'
    )
    sst_full = sst_phrases.merge(sst_labels, on='phrase ids')[['phrase', 'sentiment values']].rename({'phrase': 'review', 'sentiment values': 'sentiment'})

    imdb_data = pd.read_csv(
        '../data/imdb_utf8.csv'
    )

    full_data = DataLoader([
        (imdb_data, OrdinalEncoder(categories=[['negative', 'positive']]))
    ], 300)

    x, y, embedding_matrix, vocab_size = full_data.get_encoded_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2
    )

    model = FinSentimentModel(embedding_matrix, vocab_size)
    model.compile()
    model.fit(x_train, y_train)
    test_predictions = model.predict(x_test)

    print('Accuracy :', accuracy_score(test_predictions, y_test))
