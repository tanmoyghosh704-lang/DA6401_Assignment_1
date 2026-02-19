import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split

def one_hot_encode(labels, num_classes=10):
    
    one_hot_encoded = np.zeros((labels.size, num_classes))
    one_hot_encoded[np.arange(labels.size), labels] = 1
    return one_hot_encoded

def load_and_preprocess_data(dataset_name, val_split=0.1, random_state=42):
    
    
    
    if dataset_name.lower() == 'mnist':
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    elif dataset_name.lower() == 'fashion_mnist':
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose 'mnist' or 'fashion_mnist'.")

    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=val_split, random_state=random_state
    )

    
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    
    X_train = X_train / 255.0
    X_valid = X_valid / 255.0
    X_test = X_test / 255.0

    
    y_train = one_hot_encode(y_train, 10)
    y_valid = one_hot_encode(y_valid, 10)
    y_test = one_hot_encode(y_test, 10)

    print(f"Data Loaded Successfully: {dataset_name}")
    print(f"Train Data Shape: {X_train.shape}, Labels Shape: {y_train.shape}")
    print(f"Validation Data Shape: {X_valid.shape}, Labels Shape: {y_valid.shape}")
    print(f"Test Data Shape: {X_test.shape}, Labels Shape: {y_test.shape}")

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)