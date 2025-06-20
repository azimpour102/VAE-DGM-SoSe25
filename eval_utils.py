import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_losses(train_losses, val_losses, data_flag):
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(["Training Loss", "Validation Loss"])
    plt.title("Model Loss During the Training Process")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(data_flag + "_performance_analysis.png")
    plt.show()

def reconstruct_images(num, model, dataset):
    actual_images = []
    reconstructed_images = []
    for i in range(num):
        idx = np.random.randint(len(dataset))
        image, label = dataset[idx]
        actual_images.append(image.detach().cpu().reshape(28, 28))
        input = image.view(-1, 784).to(device)
        output = model(input)[0][0]
        reconstructed_images.append(output.detach().cpu().reshape(28, 28))

    return actual_images, reconstructed_images

def generate_images(num, model):
    images = []
    for i in range(num):
        z_sample = torch.randn(1, 128).to(device)
        x_decoded = model.decode(z_sample)
        image = x_decoded.detach().cpu().reshape(28, 28)
        images.append(image)

    return images

def plot_images(images, saving_name):
    rows = len(images)
    cols = len(images[0])

    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))

    for i in range(rows):
        for j in range(cols):
            axes[i, j].matshow(images[i][j], cmap='gray')
        # axes[1, j].matshow(actual_images[i], cmap='gray')

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(saving_name)
    plt.show()