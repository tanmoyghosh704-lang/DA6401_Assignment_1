import numpy as np
from src.ann.neural_layer import NeuralLayer
from src.ann.activation import Sigmoid, Tanh, ReLU, Softmax
from src.ann.objective_functions import MeanSquaredError, CrossEntropy
from src.ann.optimizers import get_optimizer

class NeuralNetwork:
    def __init__(self, cli_args):
        self.layers = []
        self.cli_args = cli_args
        
        
        if len(cli_args.hidden_size) > 6:
            raise ValueError(f"Assignment constraint violated: Number of hidden layers ({len(cli_args.hidden_size)}) exceeds the maximum of 6.")
            
        if any(size > 128 for size in cli_args.hidden_size):
            raise ValueError(f"Assignment constraint violated: One or more hidden layers exceed the maximum of 128 neurons. Sizes given: {cli_args.hidden_size}")
        
        input_size = 784
        output_size = 10
        
        activation_map = {
            'sigmoid': Sigmoid,
            'tanh': Tanh,
            'relu': ReLU
        }
        
        if cli_args.activation.lower() not in activation_map:
            raise ValueError(f"Unsupported activation function: {cli_args.activation}")
            
        hidden_activation = activation_map[cli_args.activation.lower()]()
        
        
        current_input_size = input_size
        for hidden_size in cli_args.hidden_size:
            layer = NeuralLayer(
                input_size=current_input_size, 
                output_size=hidden_size, 
                activation=hidden_activation,
                initialization=cli_args.weight_init
            )
            self.layers.append(layer)
            current_input_size = hidden_size
            
        
        output_layer = NeuralLayer(
            input_size=current_input_size,
            output_size=output_size,
            activation=Softmax(),
            initialization=cli_args.weight_init
        )
        self.layers.append(output_layer)
        
        
        if cli_args.loss.lower() == 'cross_entropy':
            self.loss_fn = CrossEntropy()
        elif cli_args.loss.lower() == 'mean_squared_error':
            self.loss_fn = MeanSquaredError()
        else:
            raise ValueError(f"Unsupported loss function: {cli_args.loss}")
            
        
        wd = getattr(cli_args, 'weight_decay', 0) 
        self.optimizer = get_optimizer(cli_args.optimizer, cli_args.learning_rate, wd)
        
    def forward(self, X):
        
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, y_true, y_pred):
        
        grad_output = self.loss_fn.backward(y_pred, y_true)
        
        
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
            
        
        grad_w = [layer.grad_W for layer in self.layers]
        grad_b = [layer.grad_b for layer in self.layers]
        
        return grad_w, grad_b
    
    def update_weights(self):
        
        
        self.optimizer.update(self.layers)
    
    def train(self, X_train, y_train, X_valid=None, y_valid=None, epochs=10, batch_size=16):
        
        num_samples = X_train.shape[0]
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            
            
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                
                y_pred = self.forward(X_batch)
                
                
                batch_loss = self.loss_fn.forward(y_pred, y_batch)
                epoch_loss += batch_loss * X_batch.shape[0]
                
                
                self.backward(y_batch, y_pred)
                
                
                self.update_weights()
                
            
            avg_loss = epoch_loss / num_samples
            _, train_acc = self.evaluate(X_train, y_train)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f}", end="")
            
            
            if X_valid is not None and y_valid is not None:
                val_loss, val_acc = self.evaluate(X_valid, y_valid)
                print(f" - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
            else:
                print()
                
            history['loss'].append(avg_loss)
            history['accuracy'].append(train_acc)
            
        return history
    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data. Mimics your cal_accuracy function.
        """
        y_pred = self.forward(X)
        loss = self.loss_fn.forward(y_pred, y)
        
        
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        
        return loss, accuracy