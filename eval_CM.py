import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def eval_cm(model, data_loader, output_dir):
    """
    Compute and plot the confusion matrix (in percentages) for the test set.
    Saves the figure PNG and appends summary stats to a text file in output_dir.
    """
    # Extract test data
    try:
        X_test, y_test = data_loader['test']
    except Exception:
        X_test, y_test = data_loader

    # Predict and get class indices
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Determine class names
    n_classes = y_test.shape[1]
    class_names = getattr(data_loader, 'label_names', [f"Class {i}" for i in range(n_classes)])

    # Compute confusion matrix and normalize to percentages
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    cm_percent = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=class_names)
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='.2f')
    ax.set_title('Confusion Matrix (%)')
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save figure
    model_name = getattr(model, 'name', 'model')
    fig_path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    fig.savefig(fig_path)
    plt.close(fig)

    # Append raw and percentage matrix to a text file
    stats_file = os.path.join(output_dir, 'confusion_matrix.txt')
    with open(stats_file, 'a') as f:
        f.write(f"=== {model_name} ===\n")
        f.write("Raw confusion matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nPercentage confusion matrix:\n")
        f.write(np.array2string(cm_percent, precision=2))
        f.write("\n\n")
