import os
import string
from typing import Any

import pandas as pd
import numpy as np
import re
import gensim
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.base import OneToOneFeatureMixin
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def encode_sentiment(sentiment: np.ndarray,
                     encoder: Any) -> np.ndarray:
    """
    Encode sentiment scores and scale them such that they have a max of 1,
    and min of 0

    :param sentiment: Sentiment scores
    :param encoder: Encoder
    :return: Sentiment scores with the encoder applied to them, also scaled
    to be within 0 to 1
    """
    if encoder is not None:
        encoded_sentiment = encoder.fit_transform(
            sentiment.reshape(-1, 1)
        ).astype(np.int64)
    else:
        encoded_sentiment = sentiment.reshape((-1, 1))

    encoded_sentiment = MinMaxScaler().fit_transform(encoded_sentiment)
    return encoded_sentiment


def clean_review(review: str) -> str:
    """
    Clean a review

    :param review: string to be cleaned
    :return: Cleaned review
    """

    # remove punctuation from X
    strip_punct = str.maketrans('', '', string.punctuation)
    review = review.translate(strip_punct)

    # replace double spaces with single spaces and remove spaces at the end of
    # sentences
    review = re.sub(' +', ' ', review)
    review = re.sub(' $', '', review)
    review = re.sub('^ ', '', review)

    # remove @, links, and non-alphanumeric characters
    review = re.sub('@\\S+|https?:\\S+|http?:\\S|[^A-Za-z0-9]+', ' ', review)

    # lower case everything
    review = review.lower()

    return review


def clean_reviews(reviews: np.ndarray) -> np.ndarray:
    return np.array(list(map(clean_review, list(reviews)))).reshape((-1, 1))


def validate_dataframe(dataframe: pd.DataFrame) -> None:
    """
    Ensure the dataframe has a 'review' and 'sentiment' column, and no other
    column. If any of these is violated, throw an exception.

    :param dataframe: The dataframe to validate
    """
    if 'review' not in dataframe.columns:
        raise Exception(
            'Malformed dataframe',
            dataframe,
            'doesnt contain "review" column'
        )
    elif 'sentiment' not in dataframe.columns:
        raise Exception(
            'Malformed dataframe',
            dataframe,
            'doesnt contain "sentiment" column'
        )
    elif len(dataframe.columns) != 2:
        raise Exception(
            'Malformed dataframe',
            dataframe,
            'contains more than 2 columns'
        )


class DataLoader:
    full_frame: pd.DataFrame
    encoded: bool
    max_sequence_length: int

    def __init__(
            self,
            dataframe_encoder_pairs: list[(pd.DataFrame, OneToOneFeatureMixin)],
            max_sequence_length: int,
    ):
        """
        Create a dataframe containing the reviews and sentiment scores of every
        provided dataframe by applying each encoder to its corresponding dataset
        and concatenating them along the horizontal axis.

        :param dataframe_encoder_pairs: The datasets and their corresponding
        encoders
        """
        self.max_sequence_length = max_sequence_length
        self.encoded = False

        full_frame = pd.DataFrame({
            'review': np.array([], dtype=str),
            'sentiment': np.array([], dtype=np.int32)
        })

        for dataframe, encoder in dataframe_encoder_pairs:
            # incase the dataframe uses object or some other data type
            dataframe['review'] = dataframe['review'].astype('str')

            # validate dataframe, clean the data and encode the sentiment scores
            validate_dataframe(dataframe)
            encoded_sentiment = encode_sentiment(
                dataframe['sentiment'].to_numpy(),
                encoder
            )
            cleaned_reviews = clean_reviews(dataframe['review'].to_numpy())

            # join the data into a single frame, and tack it on the full frame
            new_data = np.concatenate(
                [cleaned_reviews, encoded_sentiment],
                axis=1
            )
            full_frame = np.concatenate([full_frame, new_data])

        self.full_frame = pd.DataFrame(
            full_frame,
            columns=['review', 'sentiment']
        )

    def get_encoded_dataset(self) -> (np.ndarray, np.ndarray, np.ndarray):
        # 2d list of sentence, where each element is a string, use this to
        # train a word to vector model
        if not os.path.isfile('sentiment_model/saved_models/w2vmodel.kvmodel'):
            documents = [_text.split() for _text in self.full_frame.review]
            w2v_model = gensim.models.word2vec.Word2Vec(vector_size=300,
                                                        window=7,
                                                        min_count=1,
                                                        workers=8)
            w2v_model.build_vocab(documents)
            w2v_model.train(documents, total_examples=len(documents), epochs=32)
            w2v_model.save('sentiment_model/saved_models/w2vmodel.kvmodel')
        else:
            w2v_model = gensim.models.word2vec.Word2Vec.load('sentiment_model/saved_models/w2vmodel.kvmodel')

        # fit a tokenizer to the reviews
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.full_frame.review)

        vocab_size = len(tokenizer.word_index) + 1

        # pad the reviews
        padded_review = pad_sequences(
            tokenizer.texts_to_sequences(self.full_frame.review),
            maxlen=300
        )

        # encode sentiments
        encoder = LabelEncoder()
        encoder.fit(self.full_frame.sentiment.tolist())

        encoded_sentiment = encoder.transform(self.full_frame.sentiment.tolist())
        encoded_sentiment = encoded_sentiment

        # create embedding matrix
        embedding_matrix = np.zeros((vocab_size, 300))
        for word, i in tokenizer.word_index.items():
            if word in w2v_model.wv:
                embedding_matrix[i] = w2v_model.wv[word]

        return padded_review, encoded_sentiment, embedding_matrix, vocab_size


