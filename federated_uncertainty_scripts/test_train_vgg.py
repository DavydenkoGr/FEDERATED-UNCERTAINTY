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

from federated_uncertainty.nn.constants import ModelName
from federated_uncertainty.nn.load_models import get_model
from federated_uncertainty.optim.train import train_ensembles_w_dataloader
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

n_members = 2
n_epochs = 3
batch_size = 128
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

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

ensemble = [
    get_model(
        ModelName.VGG11,
        n_classes,
    ).to(device)
    # VGG('VGG19').to(device)
    for _ in range(n_members)
]

ensemble = train_ensembles_w_dataloader(
    ensemble,
    trainloader,
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
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(f"Model {i + 1} accuracy: {100.*correct/total:.4f}")
