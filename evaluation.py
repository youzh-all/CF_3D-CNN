import argparse

# Import model builders
from model_LSTM import build_lstm_model
from model_1DCNN import build_1dcnn_model
from model_3DCNN import build_3dcnn_model

# Import evaluation routines
from eval_metrics import eval_metrics
from eval_roc import eval_roc
from eval_CM import eval_cm
from eval_tsne import eval_tsne
from eval_IG import eval_ig


def run_all_evaluations(model_name, model, data_loader, output_dir):
    print(f"\n=== Evaluations for {model_name} ===")
    # Metrics (accuracy, precision, recall, F1, etc.)
    eval_metrics(model, data_loader, output_dir)
    # ROC curves and AUC
    eval_roc(model, data_loader, output_dir)
    # Confusion matrix
    eval_cm(model, data_loader, output_dir)
    # t-SNE visualization
    eval_tsne(model, data_loader, output_dir)
    # Integrated Gradients or other XAI
    eval_ig(model, data_loader, output_dir)


def main(models, data_path, output_dir):
    # Assume a common data loading function
    from make_dataset import get_data_loader
    data_loader = get_data_loader(data_path)

    for name in models:
        if name.lower() == 'lstm':
            model = build_lstm_model()
        elif name.lower() == '1dcnn':
            model = build_1dcnn_model()
        elif name.lower() == '3dcnn':
            model = build_3dcnn_model()
        else:
            print(f"Unknown model: {name}")
            continue

        run_all_evaluations(name, model, data_loader, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified evaluation script for LSTM, 1D-CNN, and 3D-CNN models')
    parser.add_argument('--models', nargs='+', default=['lstm', '1dcnn', '3dcnn'],
                        help='List of model names to evaluate')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation outputs')
    args = parser.parse_args()

    main(args.models, args.data_path, args.output_dir)
