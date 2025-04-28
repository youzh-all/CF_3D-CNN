import os
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


def eval_roc(model, data_loader, output_dir):
    """
    Compute and plot ROC curves and AUC for each class and micro-average.
    Saves a ROC plot PNG and appends AUC values to a text file in output_dir.
    """
    # Extract test set
    try:
        X_test, y_test = data_loader['test']
    except Exception:
        X_test, y_test = data_loader

    # Predict class probabilities
    y_pred_prob = model.predict(X_test)

    # Determine number of classes and binarize true labels
    n_classes = y_test.shape[1]
    y_true = np.argmax(y_test, axis=1)
    y_test_binarized = label_binarize(y_true, classes=range(n_classes))

    # Try to get class names; fallback to "Class {i}"
    label_names = getattr(data_loader, 'label_names', [f"Class {i}" for i in range(n_classes)])

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f"{label_names[i]} (AUC = {roc_auc[i]:.2f})")

    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_test_binarized.ravel(), y_pred_prob.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(
        fpr["micro"], tpr["micro"],
        label=f"Micro-average (AUC = {roc_auc['micro']:.2f})", linestyle="--"
    )

    # Plot diagonal line for random chance
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save plot to file
    model_name = getattr(model, 'name', 'model')
    plot_path = os.path.join(output_dir, f"roc_{model_name}.png")
    plt.savefig(plot_path)
    plt.close()

    # Append AUC values to a text file
    auc_file = os.path.join(output_dir, 'roc_auc.txt')
    with open(auc_file, 'a') as f:
        f.write(f"=== {model_name} ===\n")
        for i in range(n_classes):
            f.write(f"AUC for {label_names[i]}: {roc_auc[i]:.4f}\n")
        f.write(f"Micro-average AUC: {roc_auc['micro']:.4f}\n\n")
