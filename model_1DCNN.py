import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, GlobalMaxPooling1D
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

# Data preprocessing: Convert Z-axis to 1D
# Change input data to (n, Z, features) format
X_train_1d = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)  # (898, 26, 75
X_test_1d = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)  # (225, 26, 75)

class_num = 3
learning_rate = 0.000005

# Check shape
print(f"X_train_1d shape: {X_train_1d.shape}")
print(f"X_test_1d shape: {X_test_1d.shape}")

# Set input data shape
input_shape_1d = X_train_1d.shape[1:]  # (26, 75)
print(f"Input shape for 1D-CNN: {input_shape_1d}")

class_num = 3
learning_rate = 0.000005
# Define 1D-CNN model
model_1d = Sequential([
    Conv1D(128, kernel_size=3, activation='relu', input_shape=input_shape_1d),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu'),
    BatchNormalization(),
    GlobalMaxPooling1D(),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(class_num, activation='softmax')  # Number of classes (3)
])

# Compile model
model_1d.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history_1d = model_1d.fit(
    X_train_1d, y_train,
    validation_split=0.2,
    epochs=500,  # Example epoch number
    batch_size=16,  # Batch size
    verbose=1
)

# Print training results
print(f"Final training accuracy: {history_1d.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history_1d.history['val_accuracy'][-1]:.4f}")

# Evaluate test set performance
test_loss_1d, test_accuracy_1d = model_1d.evaluate(X_test_1d, y_test, verbose=1)
print(f"Test Loss: {test_loss_1d:.4f}")
print(f"Test Accuracy: {test_accuracy_1d:.4f}")

# Visualize training curves
plt.figure(figsize=(12, 4))

# Visualize accuracy
plt.subplot(1, 2, 1)
plt.plot(history_1d.history['accuracy'], label='Train Accuracy')
plt.plot(history_1d.history['val_accuracy'], label='Validation Accuracy')
plt.title('1D-CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Visualize loss
plt.subplot(1, 2, 2)
plt.plot(history_1d.history['loss'], label='Train Loss')
plt.plot(history_1d.history['val_loss'], label='Validation Loss')
plt.title('1D-CNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Save model
model_1d.save('model_1d.h5')


import tensorflow as tf


def build_1dcnn_model(weights_path='model_1d.h5'):
    """
    Load and return a compiled 1D-CNN model from the specified HDF5 file.

    Args:
        weights_path: Path to the saved Keras .h5 model file.
    Returns:
        A tf.keras.Model instance loaded with pretrained weights.
    """
    model = tf.keras.models.load_model(weights_path)
    return model