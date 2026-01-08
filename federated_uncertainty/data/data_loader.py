import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import medmnist
from medmnist import INFO


def get_dataset_stats(dataset_name):
    if dataset_name in INFO:
        info = INFO[dataset_name]
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        n_classes = len(info['label'])
        img_size = 32  
        padding = 4
    elif dataset_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        n_classes = 10
        img_size = 32
        padding = 4
    elif dataset_name == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        n_classes = 100
        img_size = 32
        padding = 4
    elif dataset_name == 'tiny-imagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
        n_classes = 200
        img_size = 64
        padding = 8
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return mean, std, n_classes, img_size, padding


def load_dataset(dataset_name, data_root='./data'):
    dataset_name = dataset_name.lower()
    mean, std, n_classes, img_size, padding = get_dataset_stats(dataset_name)
    if dataset_name in INFO:
        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info['python_class'])
        n_channels = info['n_channels']
        transforms_list = [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
        
        if n_channels == 1:
            transforms_list.insert(0, transforms.Grayscale(num_output_channels=3))
        
        transforms_list.append(transforms.Normalize(mean, std))
        train_transforms_list = transforms_list.copy()
        aug_idx = 1 if n_channels == 1 else 0
        train_transforms_list.insert(aug_idx, transforms.RandomCrop(img_size, padding=padding))
        train_transforms_list.insert(aug_idx + 1, transforms.RandomHorizontalFlip())

        transform_train = transforms.Compose(train_transforms_list)
        transform_test = transforms.Compose(transforms_list)

        trainset = DataClass(split='train', transform=transform_train, download=True, root=data_root)
        testset = DataClass(split='test', transform=transform_test, download=True, root=data_root)
        
        target_transform = transforms.Lambda(lambda y: torch.as_tensor(y).long().squeeze())
        trainset.target_transform = target_transform
        testset.target_transform = target_transform
        
        if hasattr(trainset, 'labels'):
            trainset.targets = trainset.labels.squeeze().tolist()
        if hasattr(testset, 'labels'):
            testset.targets = testset.labels.squeeze().tolist()

        return trainset, testset, n_classes

    transform_train = transforms.Compose([
        transforms.RandomCrop(img_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if dataset_name == 'cifar10':
        trainset = CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        testset = CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    
    elif dataset_name == 'cifar100':
        trainset = CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
        testset = CIFAR100(root=data_root, train=False, download=True, transform=transform_test)
    
    elif dataset_name == 'tiny-imagenet':
        # ./data/tiny-imagenet-200/train
        # ./data/tiny-imagenet-200/val
        tiny_root = os.path.join(data_root, 'tiny-imagenet-200')
        train_dir = os.path.join(tiny_root, 'train')
        val_dir = os.path.join(tiny_root, 'val')
        
        if not os.path.exists(train_dir):
             raise FileNotFoundError(f"Tiny-ImageNet not found at {train_dir}. Please download and extract it.")

        trainset = ImageFolder(root=train_dir, transform=transform_train)
        testset = ImageFolder(root=val_dir, transform=transform_test)
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    return trainset, testset, n_classes


def get_class_indices(dataset, n_classes):
    class_indices = {i: [] for i in range(n_classes)}
    
    targets = getattr(dataset, 'targets', None)
    if targets is None:
        if hasattr(dataset, 'labels'):
             targets = dataset.labels.squeeze().tolist()
        else:
             targets = [label for _, label in dataset]
         
    for idx, label in enumerate(targets):
        if hasattr(label, 'item'):
            label = label.item()
        elif isinstance(label, list) or (hasattr(label, '__len__') and len(label) == 1):
             label = int(label[0])
             
        class_indices[int(label)].append(idx)
        
    return class_indices