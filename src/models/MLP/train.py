from sklearn.neural_network import MLPRegressor


def train_mlp_model(X_train, y_train, activation='relu', bath_size=16,
                    learning_rate_init=0.005, max_iter=500, hidden_layer_sizes=(128,),
                    random_seed=42):
    """
    Train a MLP model with specified hyperparameters.

    Input
    -----
    X_train: DataFrame with training features
    y_train: Series with training target variable
    random_seed: Random seed for reproducibility (default is 42)

    Output
    ------
    model: Trained MLP model
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

    # Fit the model to the training data
    mlp_model.fit(X_train, y_train)

    return mlp_model
