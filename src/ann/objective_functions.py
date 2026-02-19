import numpy as np

class Loss:
    def forward(self, y_pred, y_true):
        pass
    def backward(self, y_pred, y_true):
        pass

class MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        # One-hot encode y_true if it isn't already, or ensure shapes match
        return np.mean(0.5 * (y_true - y_pred)**2)

    def backward(self, y_pred, y_true):
        # Derivative of MSE w.r.t y_pred
        return (y_pred - y_true) / y_true.shape[0]

class CrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # Clip y_pred to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Assuming y_true is one-hot encoded or handled as indices
        # If y_true is one-hot:
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def backward(self, y_pred, y_true):
        # Assuming Softmax was used previously.
        # This simplifies to (y_pred - y_true) / batch_size
        return (y_pred - y_true) / y_true.shape[0]