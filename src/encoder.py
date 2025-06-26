import numpy as np


class SimpleEncoder:
    """
    A simple state encoder that scales input states using a provided scaler.

    This encoder is responsible for normalizing or scaling observation states
    from the environment before they are fed into the neural network.

    Attributes:
        batch_size (int): The number of parallel environments.
        features (int): The number of features in the state space.
        scaler (MinMaxScaler): The scaler object used for normalization.
        prev_state (np.ndarray): The previous state observed, initialized to zeros.
    """

    def __init__(self, n_features, batch_size, scaler):
        """
        Initializes the SimpleEncoder.

        Args:
            n_features (int): The number of features in the input state.
            batch_size (int): The batch size for processing states.
            scaler (MinMaxScaler): A pre-fitted MinMaxScaler object.
        """
        self.batch_size = batch_size
        self.features = n_features
        self.scaler = scaler
        self.prev_state = np.zeros((self.batch_size, self.features))

    def reset(self, state=None):
        """
        Resets the internal state of the encoder.

        Args:
            state (np.ndarray, optional): The initial state to set. If None,
                the previous state is reset to zeros. Defaults to None.
        """
        if state is None:
            self.prev_state = np.zeros((self.batch_size, self.features))
        else:
            self.prev_state = self.forward(state)

    def forward(self, state):
        """
        Scales the input state using the provided scaler.

        Args:
            state (np.ndarray): The input state to be scaled.

        Returns:
            np.ndarray: The scaled state.
        """
        return self.scaler.transform(np.atleast_2d(state))
