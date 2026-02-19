import numpy as np

class Loss:
    def forward(self, y_pred, y_true):
        pass
    def backward(self, y_pred, y_true):
        pass

class MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        return np.mean(0.5*(y_true-y_pred)**2)
    def backward(self,y_pred,y_true):
        return (y_pred - y_true) / y_pred.shape[0]
class CrossEntropyLoss(Loss):
    def forward(self,y_pred,y_true):
        epsilon=1e-15
        y_pred=np.clip(y_pred,epsilon,1-epsilon)
        return -np.mean(np.sum(y_true*np.log(y_pred)),axis=1)
    def backward(self, y_pred, y_true):
        return (y_pred - y_true) / y_true.shape[0]