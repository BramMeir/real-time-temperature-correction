from sklearn.base import BaseEstimator, RegressorMixin
from tcn import TCN
import tensorflow as tf


class TCNRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        nb_filters=32,
        kernel_size=4,
        learning_rate=1e-3,
        dilations="1-2-4"
    ):
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.dilations = dilations
        self.model_ = None

    def build_model(self, input_shape):
        dilation_tuple = tuple(map(int, self.dilations.split("-")))   # ← STRING → TUPLE

        model = tf.keras.Sequential([
            TCN(
                nb_filters=self.nb_filters,
                kernel_size=self.kernel_size,
                dilations=dilation_tuple,
                activation='relu'
            ),
            tf.keras.layers.Dense(1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="mae"
        )
        return model

    def fit(self, X, y):
        X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))

        self.model_ = self.build_model(input_shape=X_reshaped.shape[1:])
        self.model_.fit(X_reshaped, y, epochs=10, verbose=0)
        return self

    def predict(self, X):
        X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))
        return self.model_.predict(X_reshaped).flatten()
