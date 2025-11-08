import sys
import argparse
from pathlib import Path

# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torchvision
import torchvision.transforms as transforms
import random
from torch.utils.data import DataLoader, Subset

from federated_uncertainty.nn.constants import ModelName
from federated_uncertainty.nn.load_models import get_model
from federated_uncertainty.optim.train import train_ensembles_w_local_data
from federated_uncertainty.randomness import set_all_seeds

from federated_uncertainty.models import *

seed = 1
set_all_seeds(seed)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
args = parser.parse_args()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

n_classes = 10
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

n_members = 5
fraction = 4
n_epochs = 6
lambda_ = 1.0
lr = 1e-3
criterion = nn.CrossEntropyLoss()

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

def get_class_indices(dataset):
    class_indices = {i: [] for i in range(n_classes)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    return class_indices

train_class_indices = get_class_indices(trainset)
test_class_indices = get_class_indices(testset)

def sample_indices(selected_classes, class_indices, total_samples):
    per_class = total_samples // len(selected_classes)
    selected_indices = []
    for c in selected_classes:
        # если в классе меньше элементов, чем нужно — выбираем все
        n = min(per_class, len(class_indices[c]))
        selected_indices += random.sample(class_indices[c], n)
    return selected_indices

samples_per_client = len(trainset) // fraction

client_train_loaders = []
client_test_loaders = []
client_classes = []

for i in range(n_members):
    n_client_classes = random.randint(2, 5)
    selected_classes = random.sample(range(n_classes), n_client_classes)
    client_classes.append(selected_classes)

    # Train
    train_indices = sample_indices(selected_classes, train_class_indices, samples_per_client)
    train_subset = Subset(trainset, train_indices)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Test (захардкожено в 4 раза меньше трейна)
    test_indices = sample_indices(selected_classes, test_class_indices, samples_per_client // 4)
    test_subset = Subset(testset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    client_train_loaders.append(train_loader)
    client_test_loaders.append(test_loader)

    print(f"Client {i+1}: classes {selected_classes}, train samples {len(train_indices)}, test samples {len(test_indices)}")

all_classes = list(range(n_classes))
common_train_indices = sample_indices(all_classes, train_class_indices, samples_per_client)
common_test_indices = [idx for idx, (_, label) in enumerate(testset) if label in all_classes]

common_train_loader = DataLoader(Subset(trainset, common_train_indices),
                                 batch_size=args.batch_size, shuffle=True, num_workers=2)
common_test_loader = DataLoader(Subset(testset, common_test_indices),
                                batch_size=args.batch_size, shuffle=False, num_workers=2)

print(f"\nTotal loaders created:")
print(f" - Client train loaders: {len(client_train_loaders)}")
print(f" - Client test loaders:  {len(client_test_loaders)}")
print(f" - Common loaders: 2 (train + test)")

ensemble = [
    # get_model(
    #     ModelName.VGG11,
    #     n_classes,
    # ).to(device)
    VGG('VGG19').to(device)
    for _ in range(n_members)
]

ensemble = train_ensembles_w_local_data(
    ensemble,
    client_train_loaders,
    device,
    n_epochs,
    lambda_=lambda_,
    criterion=criterion,
    lr=lr,
)

for i, model in enumerate(ensemble):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(client_test_loaders[i]):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(f"Model {i + 1} accuracy: {100.*correct/total:.4f}")
