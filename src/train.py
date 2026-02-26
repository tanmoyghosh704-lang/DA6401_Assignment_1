import argparse
import wandb
import numpy as np
import json
import os
from src.ann.neural_network import NeuralNetwork
from src.utils.data_loader import load_and_preprocess_data

def main():
    
    parser = argparse.ArgumentParser(description="Train a Multi-Layer Perceptron")
    
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', choices=['mnist', 'fashion_mnist'], help="Choose between mnist and fashion_mnist")
    parser.add_argument('-e', '--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('-b', '--batch_size', type=int, default=16, help="Mini-batch size")
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'], help="Choice of mean_squared_error or cross_entropy")
    parser.add_argument('-o', '--optimizer', type=str, default='nadam', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], help="Optimizer choice")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="Initial learning rate")
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0, help="Weight decay for L2 regularization")
    parser.add_argument('-nhl', '--num_layers', type=int, default=3, help="Number of hidden layers")
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128, 128, 128], help="Number of neurons in each hidden layer (list of values)")
    parser.add_argument('-a', '--activation', type=str, default='sigmoid', choices=['sigmoid', 'tanh', 'relu'], help="Choice of activation function")
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier', choices=['random', 'xavier'], help="Choice of weight initialization")
    
    args = parser.parse_args()

   
    run_name = f"hiden_ac-{args.activation}_hs-{args.num_layers}_epc-{args.epochs}_hl-{args.hidden_size[0]}_wd-{args.weight_decay}_learning_rate-{args.learning_rate}_opt-{args.optimizer}_bs-{args.batch_size}_wi-{args.weight_init}"
    
    
    
    wandb.init(project="DA6401_Assignment_1-src", name=run_name, config=vars(args))
    
    
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_and_preprocess_data(
        dataset_name=args.dataset, 
        val_split=0.1
    )
    
   
    print("Building Neural Network...")
    model = NeuralNetwork(args)
    
    
    num_samples = X_train.shape[0]
    
    for epoch in range(args.epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0
        
        for i in range(0, num_samples, args.batch_size):
            X_batch = X_shuffled[i:i + args.batch_size]
            y_batch = y_shuffled[i:i + args.batch_size]
            
            y_pred = model.forward(X_batch)
            batch_loss = model.loss_fn.forward(y_pred, y_batch)
            epoch_loss += batch_loss * X_batch.shape[0]
            
            model.backward(y_batch, y_pred)
            
            first_layer_grad_norm = np.linalg.norm(model.layers[0].grad_W)

            model.update_weights()
            
        
        train_loss = epoch_loss / num_samples
        _, train_acc = model.evaluate(X_train, y_train)
        val_loss, val_acc = model.evaluate(X_valid, y_valid)
        
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {train_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_acc * 100,
            'val_accuracy': val_acc * 100,
            'first_layer_grad_norm': first_layer_grad_norm 
        })

    
    print("Saving model weights and configuration...")
    
    
    weights_dict = {}
    for i, layer in enumerate(model.layers):
        weights_dict[f'W{i+1}'] = layer.W
        weights_dict[f'b{i+1}'] = layer.b
        
    np.save('best_model.npy', weights_dict)
    
    with open('best_config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    print("Training Complete! best_model.npy and best_config.json have been saved.")
    wandb.finish()

if __name__ == "__main__":
    main()