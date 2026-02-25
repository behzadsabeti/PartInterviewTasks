import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)


def get_transforms(augment=True):
    norm = transforms.Normalize(MEAN, STD)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm,
    ] if augment else [transforms.ToTensor(), norm])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])
    return train_tf, test_tf


def get_dataloaders(data_dir="./data", batch_size=128, val_split=0.1,
                   augment=True, num_workers=2):
    train_tf, test_tf = get_transforms(augment)
    full = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    n_val = int(len(full) * val_split)
    train_set, val_set = random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_set, shuffle=True,  **kw),
        DataLoader(val_set,   shuffle=False, **kw),
        DataLoader(test,      shuffle=False, **kw),
    )
