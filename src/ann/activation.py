import numpy as np

class Activation:
    def forward(self, x):
        pass
    def backward(self, x, grad_output):
        pass

class Sigmoid(Activation):
    def forward(self, x):
        
        return 1 / (1 + np.exp(-x))

    def backward(self, x, grad_output):
        
        sig = self.forward(x)
        return grad_output * (sig * (1 - sig))

class ReLU(Activation):
    def forward(self, x):
        
        return np.maximum(0, x)

    def backward(self, x, grad_output):
        
        der_relu = np.where(x <= 0, 0, 1)
        return grad_output * der_relu

class Tanh(Activation):
    def forward(self, x):
        
        return np.tanh(x)

    def backward(self, x, grad_output):
        
        der_tanh = 1 - (np.tanh(x) ** 2)
        return grad_output * der_tanh

class Softmax(Activation):
    def forward(self, x):
        
        x = x - np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def backward(self, x, grad_output):
        
        return grad_output