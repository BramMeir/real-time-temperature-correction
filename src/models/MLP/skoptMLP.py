from sklearn.neural_network import MLPRegressor
from ast import literal_eval


class SkoptMLP(MLPRegressor):
    """
    A minimal subclass of MLPRegressor to make it compatible with skopt
    when using tuples in a Categorical search space.

    It overrides `set_params` to convert the string representation of
    hidden_layer_sizes from the search space back into the tuple that
    MLPRegressor expects.
    """
    def set_params(self, **params):
        # Check if the special parameter 'hidden_layer_sizes' is being set
        if 'hidden_layer_sizes' in params:
            hls_param = params['hidden_layer_sizes']

            # If it's a string (from skopt), convert it to a tuple
            if isinstance(hls_param, str):
                params['hidden_layer_sizes'] = literal_eval(hls_param)

        # Call the original set_params with the corrected parameters
        return super().set_params(**params)
