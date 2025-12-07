import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy
import sys
import argparse
from pathlib import Path

# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated_uncertainty.nn import VGG, QuantVGG

def print_model_size(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / 1e6
    print(f"{label} size: {size_mb:.2f} MB")
    os.remove("temp.p")

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 50 == 0:
            print(f"Step {i}/{len(loader)}, Loss: {loss.item():.4f}")

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    BATCH_SIZE = 32
    EPOCHS = 1
    LEARNING_RATE = 0.01
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print("\n--- 1. Training FP32 Model ---")
    model = QuantVGG("VGG11", n_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_one_epoch(model, trainloader, optimizer, criterion, DEVICE)
    
    fp32_acc = evaluate(model, testloader, DEVICE)
    print(f"FP32 Accuracy: {fp32_acc:.2f}%")
    print_model_size(model, "FP32 Model")

    print("\n--- 2. Starting Quantization ---")
    model.cpu()
    model.eval()
    model.fuse_model()
    backend = "fbgemm" 
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    model_prepared = torch.quantization.prepare(model)
    print("Calibrating on test set (first 10 batches)...")
    with torch.no_grad():
        for i, (images, _) in enumerate(testloader):
            if i >= 10: break
            model_prepared(images)
    quantized_model = torch.quantization.convert(model_prepared)
    

    print("\n--- 3. Evaluation ---")
    print("Evaluating Quantized Model...")

    int8_acc = evaluate(quantized_model, testloader, device="cpu")
    
    print(f"FP32 Accuracy: {fp32_acc:.2f}%")
    print(f"Int8 Accuracy: {int8_acc:.2f}%")
    print(f"Accuracy Drop: {fp32_acc - int8_acc:.2f}%")
    

if __name__ == "__main__":
    main()