import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


def eval_tsne(model, data_loader, output_dir, layer_name=None):
    """
    Extract features from the penultimate (or specified) layer of the model,
    apply t-SNE dimensionality reduction, plot and save 2D scatter,
    and export coordinates with labels to an Excel file.

    Args:
        model: Trained tf.keras Model.
        data_loader: Dict-like or tuple providing (X_test, y_test).
        output_dir: Directory to save plots and results.
        layer_name: Optional name of layer to use for feature extraction.
    """
    # Unpack test data
    try:
        X, y = data_loader['test']
    except Exception:
        X, y = data_loader

    # Convert one-hot to class indices if needed
    if y.ndim > 1:
        y_true = np.argmax(y, axis=1)
    else:
        y_true = y

    # Normalize input if on [0,255]
    X = X.astype('float32')
    if X.max() > 1.0:
        X = X / 255.0

    # Reshape to model input shape
    input_shape = model.input_shape[1:]
    X = X.reshape((X.shape[0],) + tuple(input_shape))

    # Warm-up model to build weights
    _ = model(X[:1])

    # Select feature layer output
    if layer_name:
        feat_output = tf.keras.Model(inputs=model.inputs,
                                     outputs=model.get_layer(layer_name).output).predict(X)
    else:
        feat_output = tf.keras.Model(inputs=model.inputs,
                                     outputs=model.layers[-2].output).predict(X)

    # Flatten features
    feats_flat = feat_output.reshape((feat_output.shape[0], -1))

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, learning_rate=500)
    tsne_result = tsne.fit_transform(feats_flat)

    # Class names
    classes = getattr(data_loader, 'label_names', None)
    if classes is None:
        classes = [f"Class {i}" for i in np.unique(y_true)]

    # Plot
    cmap = ListedColormap(['#007bff', '#ffe119', '#008000'])
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1],
                          c=y_true, cmap=cmap, s=10, alpha=0.7)
    handles, labels = scatter.legend_elements()
    legend_labels = [classes[int(l)] for l in labels]
    plt.legend(handles, legend_labels, title='Classes')
    plt.title('t-SNE Visualization of Features')
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    model_name = getattr(model, 'name', 'model')
    plot_path = os.path.join(output_dir, f"tsne_{model_name}.png")
    plt.savefig(plot_path)
    plt.close()

    # Save results to Excel
    tsne_df = pd.DataFrame({
        'tsne_1': tsne_result[:, 0],
        'tsne_2': tsne_result[:, 1],
        'label': y_true
    })
    excel_path = os.path.join(output_dir, f"tsne_results_{model_name}.xlsx")
    tsne_df.to_excel(excel_path, index=False)
