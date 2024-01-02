import numpy as np
import keras


class FinSentimentModel:
    def __init__(self, embedding_matrix: np.ndarray, vocab_size: int):
        input_layer = keras.layers.Input(shape=(300,))
        my_embed_layer = keras.layers.Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=300, trainable=False)(input_layer)
        dropout_layer_1 = keras.layers.Dropout(0.5)(my_embed_layer)

        conv_11 = keras.layers.Conv1D(50, kernel_size=3, padding='same', kernel_initializer='he_uniform')(dropout_layer_1)
        max_pool_1 = keras.layers.MaxPool1D(padding='same')(conv_11)

        conv_21 = keras.layers.Conv1D(50, kernel_size=3, padding='same', kernel_initializer='he_uniform')(dropout_layer_1)
        max_pool_2 = keras.layers.MaxPool1D(padding='same')(conv_21)

        concat = keras.layers.concatenate([max_pool_1, max_pool_2], axis=1)
        dropout_layer_2 = keras.layers.Dropout(0.15)(concat)

        gru = keras.layers.GRU(128)(dropout_layer_2)
        dense = keras.layers.Dense(400)(gru)
        dropout_layer_3 = keras.layers.Dropout(0.1)(dense)
        out = keras.layers.Dense(1, activation='sigmoid')(dropout_layer_3)

        self.model = keras.models.Model(inputs=input_layer, outputs=out)

    def compile(self):
        self.model.compile(loss="binary_crossentropy", optimizer="adam",
                           metrics=["accuracy"])

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
        self.model.save('../saved_models/sentiment.keras')

    def predict(self, predict_data):
        predictions = self.model.predict(predict_data)
        discrete_preds = np.zeros((len(list(predictions)), 1))
        discrete_preds[predictions <= 1/2] = 0
        discrete_preds[predictions > 1/2] = 1
        return discrete_preds
