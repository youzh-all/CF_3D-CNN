import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


def integrated_gradients(model, baseline, input_data, target_class_idx, steps=50):
    """
    Compute Integrated Gradients for a single input sample and target class.
    """
    # Generate interpolated inputs between baseline and the sample
    interpolated = [baseline + (float(i) / steps) * (input_data - baseline) for i in range(steps + 1)]
    interpolated = tf.convert_to_tensor(np.stack(interpolated), dtype=tf.float32)

    # Compute gradients
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        preds = model(interpolated)
        target = preds[:, target_class_idx]
    grads = tape.gradient(target, interpolated)

    # Approximate integral
    avg_grads = tf.reduce_mean(grads[:-1] + grads[1:], axis=0) / 2.0
    ig = (input_data - baseline) * avg_grads
    return ig.numpy()


def eval_ig(model, data_loader, output_dir, steps=50):
    """
    Run Integrated Gradients on the first test sample, plot feature importances,
    and save results.
    """
    # Unpack test data
    try:
        X_test, y_test = data_loader['test']
    except Exception:
        X_test, y_test = data_loader

    # Baseline: mean of test inputs
    baseline = np.mean(X_test, axis=0)

    # Select first sample
    input_sample = X_test[0]

    # Warm-up
    _ = model(input_sample[np.newaxis, ...])

    # Predict and get target class
    preds = model.predict(input_sample[np.newaxis, ...])
    target_class = np.argmax(preds, axis=1)[0]

    # Compute IG
    ig_vals = integrated_gradients(model, baseline, input_sample, target_class, steps=steps)

    # Aggregate importance by summing absolute values over all but the first axis
    ig_axes = tuple(range(1, ig_vals.ndim))
    importance = np.sum(np.abs(ig_vals), axis=ig_axes)

    # Plot bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), importance)
    plt.xlabel('Feature Index')
    plt.ylabel('Integrated Gradients Importance')
    plt.title('Feature Importance via Integrated Gradients')
    plt.tight_layout()

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    model_name = getattr(model, 'name', 'model')
    plot_path = os.path.join(output_dir, f"ig_{model_name}.png")
    plt.savefig(plot_path)
    plt.close()

    # Save importance values to Excel
    df = pd.DataFrame({'IG_importance': importance})
    excel_path = os.path.join(output_dir, f"ig_{model_name}.xlsx")
    df.to_excel(excel_path, index=False)
