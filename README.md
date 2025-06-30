# VAE-DGM Project

This project implements Variational Autoencoders (VAE) for medical image datasets using MedMNIST.

## Setup

### Option 1: Using Conda (Recommended)

#### For GPU users (with CUDA):
```bash
conda env create -f environment.yml
conda activate vae-dgm
```

#### For CPU-only users:
```bash
conda env create -f environment_cpu.yml
conda activate vae-dgm-cpu
```

### Option 2: Using pip
```bash
pip install -r requirements.txt
```

## Running the Project

### Option 1: Using main.py (Simplified)
```bash
python main.py --dataset octmnist --batch_size 128 --learning_rate 0.001 --epochs 5
```

### Option 2: Using train.py (Full training with config)
```bash
python train.py --dataset bloodmnist --batch_size 128 --epochs 10 --learning_rate 0.001 --model_name ConvolutionalVAE --model_path bloodmnistConvolutionalVAE.pth
```

### Option 3: Using the shell script
```bash
bash run.sh
```

## Available Datasets
- `octmnist`: OCT images
- `bloodmnist`: Blood cell images
- `pathmnist`: Pathology images
- `dermamnist`: Dermatology images
- `chestmnist`: Chest X-ray images

## Available Models
- `FullyConnectedVAE`: Fully connected VAE for flattened images
- `ConvolutionalVAE`: Convolutional VAE for 2D images

## Command Line Arguments

- `--dataset`: Dataset to use (default: octmnist)
- `--batch_size`: Batch size for training (default: 128)
- `--epochs`: Number of training epochs (default: 5)
- `--learning_rate`: Learning rate (default: 0.001)
- `--model_name`: Model type (FullyConnectedVAE or ConvolutionalVAE)
- `--model_path`: Path to save/load model
- `--load_model`: Flag to load existing model

## Output
The training will generate:
- Loss plots saved as `{dataset}_performance_analysis.png`
- Reconstructed images saved as `{dataset}_reconstruction.png`
- Generated images saved as `{dataset}_generation.png`
- Model checkpoint saved as `{dataset}{model_name}.pth` 