# See Transformer-based deep learning architecture for time series forecasting
# Reference: https://www.sciencedirect.com/science/article/pii/S2665963824001040
import os
from tensorflow import keras
from keras import layers
import tensorflow as tf
from keras.saving import register_keras_serializable

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@register_keras_serializable(package="Custom", name="Time2Vec")
class Time2Vec(layers.Layer):
    """
    Time2Vec or Sinusoidal Positional Encoding.
    For simplicity in time series, we often use fixed Sin/Cos encoding
    to help the model understand the order of data points.
    """
    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim

    def call(self, inputs):
        # Create positions [0, 1, ..., seq_len-1]
        positions = tf.range(start=0, limit=self.sequence_length, delta=1, dtype=tf.float32)

        # Calculate angle rates
        d_model = tf.cast(self.embed_dim, tf.float32)
        angle_rads = positions[:, tf.newaxis] / tf.pow(10000.0, (2 * (tf.range(d_model)[tf.newaxis, :] // 2)) / d_model)

        # Apply sin to even indices, cos to odd indices
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)

        # Add batch dimension so it matches input shape
        return inputs + pos_encoding[tf.newaxis, :, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "embed_dim": self.embed_dim,
        })
        return config


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    A single Transformer encoder block.
    """
    # Normalizes the inputs
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)

    # Multi-Head Self Attention
    # Every head computes attention scores that show how much focus to put on the other time steps
    # After this layer, each time step's representation is enriched with information from other time steps
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)

    x = layers.Dropout(dropout)(x)

    # Residual connection
    # Add the input back to the output of the attention layer, this to prevent information loss
    res = x + inputs

    # Feed Forward network
    # A simple fully connected feed-forward network applied to each time step independently
    # This non-linear transformation helps the model learn complex patterns
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation="gelu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)

    # Another residual connection after the feed-forward network
    return x + res


def build_bayes_transformer_model(hp, num_features, sequence_length):
    """
    Build and compile a Transformer model based on hyperparameters.
    """
    inputs = keras.Input(shape=(sequence_length, num_features))

    # Hyperparameters
    embed_dim = hp.Int("embed_dim", min_value=64, max_value=512, step=64)
    num_heads = hp.Int("num_heads", min_value=2, max_value=6, step=2)
    ff_dim = hp.Int("ff_dim", min_value=4, max_value=256, step=16)
    num_blocks = hp.Int("num_blocks", min_value=2, max_value=6, step=1)
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.2, step=0.1)

    x = layers.Dense(embed_dim)(inputs)

    # Add positional encoding
    x = Time2Vec(sequence_length, embed_dim)(x)

    # Stack Transformer blocks
    for _ in range(num_blocks):
        x = transformer_encoder(
            x,
            head_size=embed_dim // num_heads,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout_rate
        )

    # (batch_size, sequence_length, embed_dim)
    # Decide which time steps are most important
    x = layers.Attention()([x, x])

    # Final pooling by averaging over time steps
    # Each timestep contributes proportionally to its importance
    x = layers.GlobalAveragePooling1D()(x)

    # Map the summarized representation to the output scalar value
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="mse",
        metrics=["mae"]
    )

    return model


def build_transformer_model(embed_dim, num_heads, ff_dim, num_blocks, dropout_rate, num_features, sequence_length):
    """
    Build and compile a Transformer model based on hyperparameters.
    """
    # (batch_size, sequence_length, num_features)
    inputs = keras.Input(shape=(sequence_length, num_features))

    # Feature projection
    # This layer projects input features to the embedding dimension
    # (batch_size, sequence_length, embed_dim)
    x = layers.Dense(embed_dim)(inputs)

    # Add positional encoding
    # This adds time-based positional information to the embeddings, helping the model understand the order of data points
    x = Time2Vec(sequence_length, embed_dim)(x)

    # Stack Transformer blocks
    for _ in range(num_blocks):
        x = transformer_encoder(
            x,
            head_size=embed_dim // num_heads,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout_rate
        )

    # Decode and Output
    # Add an attention layer before pooling to let the model focus on important time steps
    x = layers.Attention()([x, x])
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="mse",
        metrics=["mae"]
    )

    return model
