import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

# Load data
X = np.load('dataset/X.npy')
y = np.load('dataset/y.npy')  

# Convert string labels to numeric labels
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y)

# Check numeric labels
print(f"Numeric labels: {y_numeric[:10]}")  # Example: [0, 1, 2, ...]

num_classes = len(np.unique(y_numeric))
# One-hot encoding
y_onehot = to_categorical(y_numeric, num_classes=num_classes)

# Check the result
print(f"y_onehot shape: {y_onehot.shape}")  # (1400, 10)

# Split training and testing data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Normalize data (convert to [0, 1] range)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Check normalized data
print(f"X_train min: {X_train.min():.4f}, max: {X_train.max():.4f}")
print(f"X_test min: {X_test.min():.4f}, max: {X_test.max():.4f}")

# Check shape
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")



# Data transformation: (n_samples, 26, 5, 5, 3) -> (n_samples, n_features)
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # (898, 26*5*5*3)
X_test_flat = X_test.reshape(X_test.shape[0], -1)  # (225, 26*5*5*3)

# Check data shape after flattening
print(f"Flattened X_train shape: {X_train_flat.shape}")
print(f"Flattened X_test shape: {X_test_flat.shape}")

# SVM model definition
svm_model = SVC(kernel='rbf', probability=True, random_state=42)  # Use RBF kernel

# Model training
svm_model.fit(X_train_flat, np.argmax(y_train, axis=1))  # y_train is one-hot encoded, so argmax is needed

# Save model
svm_model.save('model_svm.h5')


import tensorflow as tf


def build_svm_model(weights_path='model_svm.h5'):
    """
    Load and return a compiled SVM model from the specified HDF5 file.

    Args:
        weights_path: Path to the saved Keras .h5 model file.
    Returns:
        A tf.keras.Model instance loaded with pretrained weights.
    """
    model = tf.keras.models.load_model(weights_path)
    return model