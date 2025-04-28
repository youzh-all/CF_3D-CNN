import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def eval_metrics(model, data_loader, output_dir):
    """
    Evaluate the given model on test data and compute loss, accuracy, precision, recall, and F1-score.
    Results are printed and appended to a metrics file in output_dir.
    """
    # Unpack test set (assumes data_loader returns a tuple or dict)
    try:
        # If data_loader is a dict with 'test' key
        X_test, y_test = data_loader['test']
    except Exception:
        # Otherwise assume a simple tuple
        X_test, y_test = data_loader

    # Evaluate loss and accuracy
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

    # Generate predictions and class labels
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Compute precision, recall, and F1-score
    precision = precision_score(y_true_classes, y_pred_classes, average='macro')
    recall = recall_score(y_true_classes, y_pred_classes, average='macro')
    f1 = f1_score(y_true_classes, y_pred_classes, average='macro')

    # Print metrics
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Append results to a text file
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    model_id = getattr(model, 'name', 'model')
    with open(metrics_path, 'a') as f:
        f.write(f"=== {model_id} ===\n")
        f.write(f"Test Loss: {loss:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
