import numpy as np
import pandas as pd
import api.sentiment_model.preprocessing.data_loader as dl
from sklearn.preprocessing import OrdinalEncoder

class TestDataLoader:
    sample_reviews = pd.DataFrame(np.array([
        [' This is a sample"s. yo D   '],
        ['This is also a sample'],
        [' This is a sample"s. yo D   '],
        ['This is also a sample']
    ]))

    sample_sentiments = pd.DataFrame(np.array([
        ['positive'],
        ['negative'],
        ['neutral'],
        ['positive']
    ]))

    def test_encode_sentiments(self):
        truth_matrix = dl.encode_sentiment(
            self.sample_sentiments,
            OrdinalEncoder(categories=[['negative', 'neutral', 'positive']])
        ) == np.array([[1], [0], [0.5], [1]])
        assert np.logical_and.reduce(truth_matrix)[0]

    def test_clean_reviews(self):
        truth_matrix = dl.clean_reviews(
            self.sample_reviews
        ) == np.array([
            ['this is a samples yo d'],
            ['this is also a sample'],
            ['this is a samples yo d'],
            ['this is also a sample']
        ])
        assert np.logical_and.reduce(truth_matrix)[0]

    def test_data_loader(self):
        data_loader = dl.DataLoader([
            (
                pd.concat([self.sample_reviews, self.sample_sentiments], axis=1),
                OrdinalEncoder(categories=[['negative', 'neutral', 'positive']])
            ),
            (
                pd.concat([self.sample_reviews, self.sample_sentiments], axis=1),
                OrdinalEncoder(categories=[['negative', 'neutral', 'positive']])
            )
        ])
        truth_matrix = np.logical_and.reduce(data_loader.full_frame == np.array([
            ['this is a samples yo d', 1],
            ['this is also a sample', 0],
            ['this is a samples yo d', 0.5],
            ['this is also a sample', 1],
            ['this is a samples yo d', 1],
            ['this is also a sample', 0],
            ['this is a samples yo d', 0.5],
            ['this is also a sample', 1]
        ]))
        assert truth_matrix[0]


if __name__ == '__main__':
    # pytest.main(['data_loader_test.py'])
    sample_reviews = pd.DataFrame(np.array([
        [' This is a sample"s. yo D   '],
        ['This is also a sample'],
        [' This is a sample"s. yo D   '],
        ['This is also a sample']
    ]))

    sample_sentiments = pd.DataFrame(np.array([
        ['positive'],
        ['negative'],
        ['neutral'],
        ['positive']
    ]))

    data_loader = dl.DataLoader([
        (
            pd.concat([sample_reviews, sample_sentiments], axis=1),
            OrdinalEncoder(categories=[['negative', 'neutral', 'positive']])
        )
    ])
    for i in data_loader.encode():
        print(i)
