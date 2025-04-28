#%%
import os
import numpy as np
import pandas as pd

# Set the data folder path
data_folder = "ROI_3D_CF"

# Create a dictionary to store class counts
class_counts = {}

# Check the npy files in the data folder
for file_name in os.listdir(data_folder):
    if file_name.endswith(".npy"):
        # Extract the class name
        class_name = file_name.split("_")[-2]
        
        # Increase the count for each class
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1

# Print the number of files for each class
print("\nNumber of data for each class:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}개")

# Create lists for X and y data storage
X = []
y = []

# List of classes to select
selected_classes = ['TripleD', 'ga20ox2', 'ga20ox4']

# Check the npy files in the data folder
for file_name in os.listdir(data_folder):
    if file_name.endswith(".npy"):
        # Create a file path
        file_path = os.path.join(data_folder, file_name)
        
        # Load the npy file
        data = np.load(file_path)
        
        # Extract the class name
        class_name = file_name.split("_")[-2]
        
        # Add only the selected classes
        if class_name in selected_classes:
            X.append(data)
            y.append(class_name)

# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"X shape: {X.shape}\ny shape: {y.shape}")
print(f"Classes: {np.unique(y)}")

# Save the dataset
np.save(os.path.join("dataset", "X.npy"), X)
np.save(os.path.join("dataset", "y.npy"), y)

# Print dataset information
print(f"Dataset created:\nX shape: {X.shape}\ny shape: {y.shape}")
