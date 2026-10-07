from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_mlp_model(X_train, y_train, activation='relu', bath_size=16,
                    learning_rate_init=0.005, max_iter=500, hidden_layer_sizes=(128,),
                    random_seed=42):
    """
    Train a MLP model with specified hyperparameters.

    Features and target are standardized with the statistics of the training window, so the network
    works on values around zero instead of raw temperatures. predict() still takes and returns
    degrees, so the recursive forecast can keep feeding its own predictions back into the lags.

    Input
    -----
    X_train: DataFrame with training features
    y_train: Series with training target variable
    random_seed: Random seed for reproducibility (default is 42)

    Output
    ------
    model: Trained MLP model, wrapped with its scalers
    """
    # Initialize the base MLP model
    mlp_model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        batch_size=bath_size,
        learning_rate_init=learning_rate_init,
        random_state=random_seed,
        solver="adam",
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1,
        max_iter=max_iter,
    )

    # Standardize each feature and the target with the statistics of the training window
    model = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), mlp_model),
                                       transformer=StandardScaler())

    # Fit the model to the training data
    model.fit(X_train, y_train)

    return model
