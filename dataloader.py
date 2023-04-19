import os
import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision import transforms

class CIFAR5M(torchvision.datasets.VisionDataset):
    def __init__(self, root, subset_len=None, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.data = []
        self.targets = []

        for ite in range(6):
            file_path = os.path.join(self.root, 'cifar5m_part%d.npz' % ite)
            data = np.load(file_path, 'r')

            self.data.extend(data["X"])
            self.targets.extend(data["Y"])

            if subset_len is not None and len(self.data) > subset_len:
                self.data = self.data[:subset_len]
                self.targets = self.targets[:subset_len]
                break

        self.data = np.array(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)




def get_cifar10_data(batch_size, subset=None):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    if subset is not None:
        subset_len = int(subset*len(trainset))
        trainset, _ = torch.utils.data.random_split(trainset, [subset_len, len(trainset)-subset_len], generator=torch.Generator().manual_seed(0))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader


def get_cifar5m_data(batch_size, subset=50000):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR5M(root = './data/cifar-5m/', subset_len=subset, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader
