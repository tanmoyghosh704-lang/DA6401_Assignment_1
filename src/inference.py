import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.ann.neural_network import NeuralNetwork
from src.utils.data_loader import load_and_preprocess_data


class ConfigArgs:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def main():
    print("Loading configuration and weights...")
    
    
    try:
        with open('best_config.json', 'r') as f:
            config_dict = json.load(f)
        args = ConfigArgs(**config_dict)
    except FileNotFoundError:
        print("Error: best_config.json not found. Did the training finish successfully?")
        return

    
    try:
        
        weights = np.load('best_model.npy', allow_pickle=True).item()
    except FileNotFoundError:
        print("Error: best_model.npy not found.")
        return

    
    _, _, (X_test, y_test) = load_and_preprocess_data(
        dataset_name=args.dataset, 
        val_split=0.1
    )

    #
    print("Rebuilding Neural Network...")
    model = NeuralNetwork(args)

    
    for i, layer in enumerate(model.layers):
        weight_key = f'W{i+1}'
        bias_key = f'b{i+1}'
        
        if weight_key in weights and bias_key in weights:
            layer.W = weights[weight_key]
            layer.b = weights[bias_key]

    
    print("Running forward pass on test data...")
    y_pred_probs = model.forward(X_test)
    
    
    predictions = np.argmax(y_pred_probs, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)

    #
    print("\n" + "="*30)
    print("   FINAL INFERENCE METRICS")
    print("="*30)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()