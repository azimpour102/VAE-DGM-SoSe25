from vae_base import *
import argparse

import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Training script for VAE.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="octmnist",
        required=True,
        help="Dataset to use for training."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        required=True,
        help="Batch size for training."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        required=True,
        help="Number of epochs for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        required=True,
        help="Learning rate for training."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="FullyConnectedVAE",
        required=True,
        help="Model to use for training."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="FullyConnectedVAE",
        required=True,
        help="Model to use for training."
    )
    parser.add_argument(
        "--load_model",
        action="store_true",
        # type=bool,
        # default=False,
        # required=True,
        help="Load model from path."
    )

    return parser.parse_args()

args = parse_args()

DATASET = (args.dataset).split(',')
LOAD_MODEL = args.load_model
SAVED_MODEL_PATH = args.model_path
MODEL_NAME = args.model_name

# Enhanced model mappings with backwards compatibility
if MODEL_NAME == 'FullyConnectedVAE' or MODEL_NAME == 'ImprovedFullyConnectedVAE':
    MODEL_TYPE = ImprovedFullyConnectedVAE
elif MODEL_NAME == 'ConvolutionalVAE' or MODEL_NAME == 'ImprovedConvolutionalVAE':
    MODEL_TYPE = ImprovedConvolutionalVAE
elif MODEL_NAME == 'ConditionalConvolutionalVAE' or MODEL_NAME == 'ImprovedConditionalConvolutionalVAE':
    MODEL_TYPE = ImprovedConditionalConvolutionalVAE
else:
    # Fallback error message
    available_models = [
        'FullyConnectedVAE', 'ImprovedFullyConnectedVAE',
        'ConvolutionalVAE', 'ImprovedConvolutionalVAE', 
        'ConditionalConvolutionalVAE', 'ImprovedConditionalConvolutionalVAE'
    ]
    raise ValueError(f"Unknown model name: {MODEL_NAME}. Available models: {available_models}")

BATCH_SIZE = args.batch_size 
TRAINING_EPOCHS = args.epochs
LR = args.learning_rate 

print(f"Selected model: {MODEL_NAME} -> {MODEL_TYPE.__name__}") 