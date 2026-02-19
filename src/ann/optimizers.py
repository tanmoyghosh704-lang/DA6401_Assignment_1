import numpy as np

class Optimizer:
    def __init__(self, learning_rate, weight_decay=0):
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def update(self, layers):
        pass

class SGD(Optimizer):
    def update(self, layers):
        for layer in layers:
            
            layer.W -= self.lr * (layer.grad_W + self.weight_decay * layer.W)
            layer.b -= self.lr * layer.grad_b

class Momentum(Optimizer):
    def __init__(self, learning_rate, beta=0.9, weight_decay=0):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.velocity = {}

    def update(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.velocity:
                self.velocity[i] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}

            
            self.velocity[i]['W'] = self.beta * self.velocity[i]['W'] + layer.grad_W
            self.velocity[i]['b'] = self.beta * self.velocity[i]['b'] + layer.grad_b

            
            layer.W -= self.lr * (self.velocity[i]['W'] + self.weight_decay * layer.W)
            layer.b -= self.lr * self.velocity[i]['b']

class NAG(Optimizer):
    def __init__(self, learning_rate, beta=0.9, weight_decay=0):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.velocity = {}

    def update(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.velocity:
                self.velocity[i] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
            
            
            self.velocity[i]['W'] = self.beta * self.velocity[i]['W'] + layer.grad_W
            self.velocity[i]['b'] = self.beta * self.velocity[i]['b'] + layer.grad_b
            
            update_W = self.beta * self.velocity[i]['W'] + layer.grad_W
            update_b = self.beta * self.velocity[i]['b'] + layer.grad_b

            layer.W -= self.lr * (update_W + self.weight_decay * layer.W)
            layer.b -= self.lr * update_b

class RMSprop(Optimizer):
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-8, weight_decay=0):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.epsilon = epsilon
        self.squared_gradients = {}

    def update(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.squared_gradients:
                self.squared_gradients[i] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}

            
            self.squared_gradients[i]['W'] = self.beta * self.squared_gradients[i]['W'] + (1 - self.beta) * (layer.grad_W ** 2)
            self.squared_gradients[i]['b'] = self.beta * self.squared_gradients[i]['b'] + (1 - self.beta) * (layer.grad_b ** 2)

            
            layer.W -= self.lr * (layer.grad_W / (np.sqrt(self.squared_gradients[i]['W']) + self.epsilon) + self.weight_decay * layer.W)
            layer.b -= self.lr * (layer.grad_b / (np.sqrt(self.squared_gradients[i]['b']) + self.epsilon))

class Adam(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0):
        super().__init__(learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m_t = {}
        self.v_t = {}

    def update(self, layers):
        self.t += 1
        for i, layer in enumerate(layers):
            if i not in self.m_t:
                self.m_t[i] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
                self.v_t[i] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}

            
            self.m_t[i]['W'] = self.beta1 * self.m_t[i]['W'] + (1 - self.beta1) * layer.grad_W
            self.m_t[i]['b'] = self.beta1 * self.m_t[i]['b'] + (1 - self.beta1) * layer.grad_b
            
            self.v_t[i]['W'] = self.beta2 * self.v_t[i]['W'] + (1 - self.beta2) * (layer.grad_W ** 2)
            self.v_t[i]['b'] = self.beta2 * self.v_t[i]['b'] + (1 - self.beta2) * (layer.grad_b ** 2)
            
            m_hat_W = self.m_t[i]['W'] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m_t[i]['b'] / (1 - self.beta1 ** self.t)
            v_hat_W = self.v_t[i]['W'] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v_t[i]['b'] / (1 - self.beta2 ** self.t)
            
            layer.W -= self.lr * (m_hat_W / (np.sqrt(v_hat_W) + self.epsilon) + self.weight_decay * layer.W)
            layer.b -= self.lr * (m_hat_b / (np.sqrt(v_hat_b) + self.epsilon))

class Nadam(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0):
        super().__init__(learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m_t = {}
        self.v_t = {}

    def update(self, layers):
        self.t += 1
        for i, layer in enumerate(layers):
            if i not in self.m_t:
                self.m_t[i] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
                self.v_t[i] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}

            
            self.m_t[i]['W'] = self.beta1 * self.m_t[i]['W'] + (1 - self.beta1) * layer.grad_W
            self.m_t[i]['b'] = self.beta1 * self.m_t[i]['b'] + (1 - self.beta1) * layer.grad_b
            
            m_hat_W_lookahead = self.beta1 * self.m_t[i]['W'] + (1 - self.beta1) * layer.grad_W
            m_hat_b_lookahead = self.beta1 * self.m_t[i]['b'] + (1 - self.beta1) * layer.grad_b
            
            self.v_t[i]['W'] = self.beta2 * self.v_t[i]['W'] + (1 - self.beta2) * (layer.grad_W ** 2)
            self.v_t[i]['b'] = self.beta2 * self.v_t[i]['b'] + (1 - self.beta2) * (layer.grad_b ** 2)
            
            m_hat_W = m_hat_W_lookahead / (1 - self.beta1 ** self.t)
            m_hat_b = m_hat_b_lookahead / (1 - self.beta1 ** self.t)
            v_hat_W = self.v_t[i]['W'] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v_t[i]['b'] / (1 - self.beta2 ** self.t)
            
            layer.W -= self.lr * (m_hat_W / (np.sqrt(v_hat_W) + self.epsilon) + self.weight_decay * layer.W)
            layer.b -= self.lr * (m_hat_b / (np.sqrt(v_hat_b) + self.epsilon))


def get_optimizer(optimizer_name, learning_rate, weight_decay=0):
    name = optimizer_name.lower()
    if name == 'sgd':
        return SGD(learning_rate, weight_decay)
    elif name == 'momentum':
        return Momentum(learning_rate, weight_decay=weight_decay)
    elif name == 'nag':
        return NAG(learning_rate, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return RMSprop(learning_rate, weight_decay=weight_decay)
    elif name == 'adam':
        return Adam(learning_rate, weight_decay=weight_decay)
    elif name == 'nadam':
        return Nadam(learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")