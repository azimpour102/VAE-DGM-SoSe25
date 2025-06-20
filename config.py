from vae_base import *
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DATASETS = [
    # 'bloodmnist',
    'octmnist',
    # 'pathmnist',
    # 'chestmnist',
]

LOAD_MODEL = True
SAVED_MODEL_PATH = 'bloodmnistFullyConnectedVAE.pth'

MODEL = FullyConnectedVAE
MODEL_NAME = 'FullyConnectedVAE'

BATCH_SIZE = 128
TRAINING_EPOCHS = 5
LR = 0.0001