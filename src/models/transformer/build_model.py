# https://www.scaler.com/topics/tensorflow/tensorflow-transformer/

import numpy as np
import tensorflow as tf


def positional_encoding(length, depth):
    """
    Creates a positional encoding for input sequences.
    This is necessary for the Transformer model to capture the order of the sequence and understand
    the relative positions of the time steps. Otherwise, the model would treat the input as a bag of features
    without any temporal context.

    How it works:
    Each time step in the input sequence is mapped to a point in a high-dimensional periodic space.
    Nearby time steps will have similar positional encodings, allowing the model to learn temporal relationships.

    Input
    -----
    length: Length of the input sequences (number of time steps)
    depth: Depth of the model (embedding dimension)

    Output
    ------
    A tensor of shape (1, length, depth) containing the positional encodings.
    """
    # Dividing depth by 2 to create separate sine and cosine components for each dimension (both half embedding space)
    depth = depth / 2

    # Create arrays for positions and depths.
    positions = np.arange(length)[:, np.newaxis]        # Shape: (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth    # Shape: (1, depth)

    # Calculate angle rates to be used for positional encoding.
    # This creates a unique frequency for each depth dimension, where the frequency decreases exponentially.
    angle_rates = 1 / (10000 ** depths)                 # Shape: (1, depth)

    # Calculate the angle radians for each position and depth.
    # angle_rads[pos, dim] = pos / (10000 ^ (dim / depth))
    angle_rads = positions * angle_rates                # Shape: (pos, depth)

    # Compute the positional encoding using both sine and cosine functions.
    # Concatenate sine and cosine components along the last axis.
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    # Cast the positional encoding to the TensorFlow float32 data type before returning.
    return tf.cast(pos_encoding, dtype=tf.float32)


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation="relu"),
            tf.keras.layers.Dense(d_model)
        ])

        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        attn_output = self.attention(x, x, training=training)
        x = self.norm1(x + self.dropout1(attn_output, training=training))

        ffn_output = self.ffn(x, training=training)
        x = self.norm2(x + self.dropout2(ffn_output, training=training))

        return x


class TransformerForecaster(tf.keras.Model):
    def __init__(
        self,
        *,
        num_layers,
        d_model,
        num_heads,
        dff,
        sequence_length,
        dropout_rate=0.1
    ):
        super().__init__()

        self.d_model = d_model

        # Project numeric inputs → d_model
        self.input_projection = tf.keras.layers.Dense(d_model)

        self.pos_encoding = positional_encoding(sequence_length, d_model)

        self.encoder_blocks = [
            EncoderBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # Pool over time
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()

        # Regression head
        self.out = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        # x: (batch, time, features)
        x = self.input_projection(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = x + self.pos_encoding[:tf.shape(x)[1]]
        x = self.dropout(x, training=training)

        for block in self.encoder_blocks:
            x = block(x, training=training)

        x = self.pooling(x)
        return self.out(x)


def build_transformer_model(hp, sequence_length):
    """
    Builds a Transformer model based on the provided hyperparameters.

    Input
    -----
    hp: Hyperparameters object containing the model configuration
    sequence_length: Length of the input sequences

    Output
    ------
    model: Compiled Transformer model
    """
    # Extract hyperparameters for the model architecture
    num_layers = hp.Int("num_layers", min_value=2, max_value=6, step=1)
    d_model = hp.Int("d_model", min_value=32, max_value=256, step=32)
    num_heads = hp.Int("num_heads", min_value=2, max_value=8, step=2)
    dff = hp.Int("dff", min_value=64, max_value=512, step=64)
    lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    # Create the Transformer model
    model = TransformerForecaster(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        sequence_length=sequence_length,
        dropout_rate=0.1
    )

    # Compile the model with Adam optimizer and Mean Squared Error loss
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )

    return model
