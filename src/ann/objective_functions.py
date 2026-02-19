import numpy as np

class MeanSquaredError:
    def forward(self, y_pred, y_true):
        
        return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))

    def backward(self, y_pred, y_true):
        
        m = y_true.shape[0]
        return (2 / m) * (y_pred - y_true)

class CrossEntropy:
    def forward(self, y_pred, y_true):
       
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def backward(self, y_pred, y_true):
        
        m = y_true.shape[0]
        return (y_pred - y_true) / m