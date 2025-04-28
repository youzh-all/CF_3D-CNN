import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization, GlobalMaxPooling3D
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

input_shape = X_train.shape[1:]  # Automatically get the shape of the input data
print(f"Input shape: {input_shape}")
class_num = y_train.shape[1]
print(f"Class number: {class_num}")

# 2. Model definition
model = Sequential([
    Conv3D(128, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape),
    BatchNormalization(),
    # GlobalMaxPooling3D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(class_num, activation='softmax')  # Number of classes (e.g., 10)
])

# 3. Model compilation

# Learning rate setting
learning_rate = 0.000005  # Adjust to the desired learning rate

model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Model training
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=500,  # Number of epochs (example)
    batch_size=16,  # Batch size
    verbose=1,
)

# 5. Training result output
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

# 6. Test set performance evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# 7. Learning curve visualization
plt.figure(figsize=(12, 4))

# Accuracy visualization
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Loss visualization
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Save model
model.save('model_3d.h5')


import tensorflow as tf


def build_3dcnn_model(weights_path='model_3d.h5'):
    """
    Load and return a compiled 3D-CNN model from the specified HDF5 file.

    Args:
        weights_path: Path to the saved Keras .h5 model file.
    Returns:
        A tf.keras.Model instance loaded with pretrained weights.
    """
    model = tf.keras.models.load_model(weights_path)
    return model