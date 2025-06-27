from vae_base import *
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DATASETS = [
    'bloodmnist',
    # 'octmnist',
    # 'pathmnist',
    # 'chestmnist',
]

LOAD_MODEL = True
SAVED_MODEL_PATH = 'bloodmnistConvolutionalVAE.pth'

MODEL = ConvolutionalVAE
MODEL_NAME = 'ConvolutionalVAE'

BATCH_SIZE = 128
TRAINING_EPOCHS = 0
LR = 0.0001