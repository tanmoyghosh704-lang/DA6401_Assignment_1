import numpy as np

class NeuralLayer:
    def __init__(self, input_size, output_size, activation, initialization='random'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        
        if initialization.lower() == 'xavier':
            limit = np.sqrt(2 / input_size)
            self.W = np.random.randn(input_size, output_size) * limit
            self.b = np.random.randn(1, output_size) * limit
        else: 
            self.W = np.random.randn(input_size, output_size) * 0.01
            self.b = np.random.randn(1, output_size) * 0.01
            
        
        self.input = None
        self.z = None
        
         
        self.grad_W = None
        self.grad_b = None

    def forward(self, input_data):
        self.input = input_data
        
        self.z = np.dot(self.input, self.W) + self.b
        
        return self.activation.forward(self.z)

    def backward(self, grad_output):
        
        grad_z = self.activation.backward(self.z, grad_output)
        
        m = self.input.shape[0]
        
        
        self.grad_W = np.dot(self.input.T, grad_z)
        self.grad_b = np.sum(grad_z, axis=0, keepdims=True)
        
       
        grad_input = np.dot(grad_z, self.W.T)
        
        return grad_input