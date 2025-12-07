"""
vgg_fakequant_gpu.py

Demo: train a float32 VGG on CIFAR10, then create a *quantized (fake-quant) copy* that runs on GPU
and represents a "noisy" quantized model. Both models are saved to disk as checkpoints.

Key points:
 - Uses PyTorch FakeQuant/QAT modules which run on CUDA (the fake-quant operations are
   implemented as autograd modules and execute on the device).
 - We do NOT call torch.quantization.convert (that would produce CPU-only quantized kernels).
   Instead we keep the model with fake-quant modules active and use it on GPU as the "quantized" model.
 - Workflow:
     1) Train float32 model for N1 epochs and save checkpoint.
     2) Copy float32 model -> prepare_qat (enable fake-quant modules) -> fine-tune for N2 epochs on GPU.
     3) Save fake-quant model checkpoint (this is the "noisy/quantized" model that runs on GPU).

Requirements:
  - CUDA GPU
  - torch>=1.12 (prepare_qat on GPU works in recent releases)
  - torchvision

Usage example:
  python ./federated_uncertainty_scripts/vgg_fakequant_gpu.py --epochs-fp 3 --epochs-q 2 --batch-size 128 --device cuda

Outputs:
  - model_fp.pth           (float32 trained model)
  - model_fakequant.pth    (model with fake-quant modules active; runs on GPU)

"""

import argparse
import copy
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import sys
import argparse
from pathlib import Path
# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated_uncertainty.nn import VGG

def try_fuse_model(model):
    # If user provided fuse_model, use it.
    if hasattr(model, 'fuse_model') and callable(getattr(model, 'fuse_model')):
        try:
            model.fuse_model()
            return
        except Exception:
            pass
    # Best-effort small fuser for common Conv-BN-ReLU patterns inside nn.Sequential
    import torch.quantization as tq
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            # try to fuse sequences in place
            modules = list(module.children())
            for i in range(len(modules)-2):
                if isinstance(modules[i], nn.Conv2d) and isinstance(modules[i+1], nn.BatchNorm2d) and isinstance(modules[i+2], nn.ReLU):
                    try:
                        tq.fuse_modules(module, [str(i), str(i+1), str(i+2)], inplace=True)
                    except Exception:
                        pass
        else:
            try_fuse_model(module)

# Training / evaluation helpers

def train_epoch(model, device, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start = time.time()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    elapsed = time.time() - start
    return running_loss / total, 100.0 * correct / total, elapsed


def evaluate(model, device, loader, criterion):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            l = criterion(outputs, targets)
            loss += l.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return loss / total, 100.0 * correct / total


def prepare_fakequant(model):
    import torch.quantization as tq
    # Choose a qconfig that uses FakeQuant; 'qnnpack' and 'fbgemm' are common backends. FakeQuant runs on device.
    qconfig = tq.get_default_qat_qconfig('fbgemm')
    model.qconfig = qconfig
    # fuse modules where appropriate to improve quantization (best-effort)
    try_fuse_model(model)
    tq.prepare_qat(model, inplace=True)
    return model


def main(args):
    device = args.device
    if device != 'cuda' and not args.allow_cpu:
        raise RuntimeError('This script requires CUDA GPU. Use --allow-cpu to override (not recommended).')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Create and train float32 model
    model_fp = VGG("VGG11", n_classes=10)
    model_fp = model_fp.to(device)

    optim_fp = optim.SGD(model_fp.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    print('=== Training float32 model ({} epochs) ==='.format(args.epochs_fp))
    for epoch in range(1, args.epochs_fp + 1):
        tr_loss, tr_acc, tr_time = train_epoch(model_fp, device, train_loader, optim_fp, criterion)
        te_loss, te_acc = evaluate(model_fp, device, test_loader, criterion)
        print(f'Epoch {epoch}: train_loss={tr_loss:.4f}, train_acc={tr_acc:.2f}%, test_acc={te_acc:.2f}%, time={tr_time:.1f}s')

    os.makedirs(args.output_dir, exist_ok=True)
    fp_path = os.path.join(args.output_dir, 'model_fp.pth')
    torch.save({'model_state': model_fp.state_dict()}, fp_path)
    print('Saved float32 checkpoint to', fp_path)

    # Prepare fake-quant model from the trained float model
    model_q = copy.deepcopy(model_fp)
    model_q.cpu()  # prepare_qat modifies qconfig in-place; do it on CPU then move to device
    model_q.train()
    model_q = prepare_fakequant(model_q)
    model_q = model_q.to(device)

    # Fine-tune quantized (fakequant) model for a few epochs to adapt to quantization noise
    optim_q = optim.SGD(model_q.parameters(), lr=args.lr * 0.1, momentum=0.9, weight_decay=5e-4)
    print('=== Fine-tuning fake-quant model ({} epochs) ==='.format(args.epochs_q))
    for epoch in range(1, args.epochs_q + 1):
        tr_loss, tr_acc, tr_time = train_epoch(model_q, device, train_loader, optim_q, criterion)
        te_loss, te_acc = evaluate(model_q, device, test_loader, criterion)
        print(f'Q-Epoch {epoch}: train_loss={tr_loss:.4f}, train_acc={tr_acc:.2f}%, test_acc={te_acc:.2f}%, time={tr_time:.1f}s')

    q_path = os.path.join(args.output_dir, 'model_fakequant.pth')
    # Important: save state_dict while fake-quant modules are active — this represents the noisy/quantized model.
    torch.save({'model_state': model_q.state_dict()}, q_path)
    print('Saved fake-quant checkpoint to', q_path)

    # Quick summary evaluation on GPU
    te_loss_fp, te_acc_fp = evaluate(model_fp, device, test_loader, criterion)
    te_loss_q, te_acc_q = evaluate(model_q, device, test_loader, criterion)
    print('Final results:')
    print(f'Float32 test acc: {te_acc_fp:.2f}%')
    print(f'FakeQuant test acc: {te_acc_q:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs-fp', type=int, default=3, help='Epochs to train float32 model')
    parser.add_argument('--epochs-q', type=int, default=2, help='Epochs to fine-tune fake-quant model')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--allow-cpu', action='store_true')
    parser.add_argument('--output-dir', type=str, default='outputs')
    args = parser.parse_args()
    main(args)
    