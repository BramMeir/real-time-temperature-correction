from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from tcn import TCN
from keras import backend as K
import ast


def create_tcn_model(nb_filters=32, kernel_size=3, dilations="(1, 2, 4, 8)",
                     learning_rate=0.001, input_shape=(10, 5)):
    """
    Creates and compiles a Keras model with a TCN layer for regression.

    This function is designed to be called by a scikit-learn wrapper like KerasRegressor.

    Args:
        nb_filters (int): Number of filters in the TCN layer.
        kernel_size (int): The size of the convolutional kernel.
        dilations (tuple): A tuple of dilation rates for the residual blocks.
        learning_rate (float): The learning rate for the Adam optimizer.
        input_shape (tuple): The shape of the input data (timesteps, features).

    Returns:
        A compiled Keras model.
    """
    # Convert the dilations string back to a tuple of integers
    try:
        actual_dilations = ast.literal_eval(dilations)
        nb_filters = int(nb_filters)
        kernel_size = int(kernel_size)
    except (ValueError, SyntaxError):
        raise ValueError("Dilations must be a string representation of a tuple, e.g., '(1, 2, 4)'")

    # Clear the previous Keras session to avoid clutter from old models / layers.
    K.clear_session()

    model = Sequential()

    # The input shape is defined as (timesteps, features).
    model.add(Input(shape=input_shape))

    # Add the TCN layer.
    # `return_sequences=False` ensures it outputs only the final step's prediction.
    model.add(TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        dilations=actual_dilations,
        use_skip_connections=True,
        activation='relu',
        return_sequences=False
    ))

    # A single output neuron is used for predicting the continuous temperature value.
    model.add(Dense(1))

    # Compile the model with the Adam optimizer and mean squared error for regression.
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model
