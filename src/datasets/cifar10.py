import torchvision.datasets as datasets
import torchvision.transforms
import torch
from torch.utils.data import DataLoader


def get_cifar_dataloader(batch_size, transforms, is_train=True, num_workers=2):
    dataset = datasets.CIFAR10(
                                    root='./datasets',
                                    transform=transforms,
                                    train=is_train, download=True
    )
    dataloader = DataLoader(
                                dataset,
                                batch_size=batch_size,
                                shuffle=is_train,
                                num_workers=num_workers
    )

    return dataloader
