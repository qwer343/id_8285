import torch
import torchvision.datasets as datasets
from torchvision import transforms
import torch.utils.data as data
from PIL import Image

import os
import numpy as np


def dataLoadFunc(opt):
    # Data loading parameters
    use_cuda = torch.cuda.is_available()
    params = {'batch_size': opt.batch_size, 'shuffle': True, 'num_workers': 16, 'pin_memory': True} if use_cuda else {}
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if opt.dataset in ['cifar10', 'cifar100']: 
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),normalize
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),normalize
        ])
    
    elif opt.dataset == "stl10": 
        train_transform = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),normalize
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),normalize
        ])
    elif opt.dataset == "imagenet":
        train_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transform = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
        ])
    

    if opt.dataset == 'cifar10':
        train_set = datasets.CIFAR10(root='/raid/datasets/public/cifar', train=True, download=False, transform = train_transform)
        valid_set = datasets.CIFAR10(root='/raid/datasets/public/cifar', train=False, download=False, transform = val_transform)
    elif opt.dataset == 'cifar100':
        train_set = datasets.CIFAR100(root='/raid/datasets/public/cifar', train=True, download=False, transform = train_transform)
        valid_set = datasets.CIFAR100(root='/raid/datasets/public/cifar', train=False, download=False, transform = val_transform)
    elif opt.dataset == 'imagenet':
        train_set = datasets.ImageFolder(root='/raid/datasets/public/imagenet/train', transform = train_transform)
        valid_set = datasets.ImageFolder(root='/raid/datasets/public/imagenet/val', transform = val_transform)

    elif opt.dataset == 'stl10':
        train_set = datasets.STL10(root='/raid/datasets/public/stl10', split='train', download=False, transform = train_transform)
        valid_set = datasets.STL10(root='/raid/datasets/public/stl10', split='test', download=False, transform = val_transform)
   

    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    return train_loader, valid_loader
