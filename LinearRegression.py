import numpy as np


class LinearRegression:
    """
    Linear Regression model implemented from scratch.

    Methods:
        fit(X, y): Trains the linear regression model.
        predict(X): Predicts the output using the linear model.
    """
    def __init__(self, lr=0.001, epochs=1000):
        """
        Initializes the LinearRegression model with specified learning rate and epochs.

        Args:
            lr (float, optional): The learning rate. Defaults to 0.001.
            epochs (int, optional): The number of epochs for training the model. Defaults to 1000.
        """
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fits the linear regression model to the training data.
        The method updates the weights and bias of the model.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        """
        Predicts the output for the given input data using the linear model.

        Returns:
            numpy.ndarray: The predicted values, a 1D array of shape (n_samples,).
        """
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
