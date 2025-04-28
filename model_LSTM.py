import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam
import matplotlib.pyplot as plt

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

# Data transformation: (n_samples, 26, 5, 5, 3) -> (n_samples, 26, 75)
X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)  # (898, 26, 75)
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)  # (225, 26, 75)

# Data normalization (LSTM is sensitive to data scale)
scaler = StandardScaler()
X_train_lstm = scaler.fit_transform(X_train_lstm.reshape(-1, X_train_lstm.shape[-1])).reshape(X_train_lstm.shape)
X_test_lstm = scaler.transform(X_test_lstm.reshape(-1, X_test_lstm.shape[-1])).reshape(X_test_lstm.shape)

class_num = num_classes

# LSTM model definition
model_lstm = Sequential([
    LSTM(128, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(class_num, activation='softmax')  # Number of classes (3)
])

optimizer = Adam(learning_rate=0.000005)

# Model compilation
model_lstm.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
history_lstm = model_lstm.fit(
    X_train_lstm, y_train,
    validation_split=0.2,
    epochs=500,  # Number of epochs (example)
    batch_size=16,
    verbose=1
)

# Test data evaluation
test_loss_lstm, test_accuracy_lstm = model_lstm.evaluate(X_test_lstm, y_test, verbose=1)
print(f"LSTM Test Loss: {test_loss_lstm:.4f}")
print(f"LSTM Test Accuracy: {test_accuracy_lstm:.4f}")

# Learning curve visualization

plt.figure(figsize=(12, 4))
# Accuracy visualization
plt.subplot(1, 2, 1)
plt.plot(history_lstm.history['accuracy'], label='Train Accuracy')
plt.plot(history_lstm.history['val_accuracy'], label='Validation Accuracy')
plt.title('LSTM Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Loss visualization
plt.subplot(1, 2, 2)
plt.plot(history_lstm.history['loss'], label='Train Loss')
plt.plot(history_lstm.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Save model
model_lstm.save('model_lstm.h5')


import tensorflow as tf


def build_lstm_model(weights_path='model_lstm.h5'):
    """
    Load and return a compiled LSTM model from the specified HDF5 file.

    Args:
        weights_path: Path to the saved Keras .h5 model file.
    Returns:
        A tf.keras.Model instance loaded with pretrained weights.
    """
    model = tf.keras.models.load_model(weights_path)
    return model
