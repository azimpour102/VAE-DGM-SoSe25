from medmnist import INFO, Evaluator

def get_datasets(data_flag, batch_size=128, size=28, download=True):
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])
    data_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = DataClass(split='train', transform=data_transform, size=size, download=download)
    val_dataset = DataClass(split='val', transform=data_transform, size=size, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, size=size, download=download)

    return train_dataset, test_dataset, val_dataset

def get_dataloaders(train_dataset, test_dataset, val_dataset, batch_size=128):
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=2*batch_size, shuffle=False)

    return train_loader, test_loader, val_loader