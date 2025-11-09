import sys
import argparse
from pathlib import Path

# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
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

n_models = 30
n_clients = 5
LAMBDA = 0.1 # для Uncertainty-Aware
fraction = 4
n_epochs = 5
lr = 1e-3
ensemble_selection_size = 3
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
    per_class = total_samples // len(selected_classes) if len(selected_classes) > 0 else 0
    selected_indices = []
    for c in selected_classes:
        n = min(per_class, len(class_indices[c]))
        selected_indices += random.sample(class_indices[c], n)
    return selected_indices

samples_per_client = len(trainset) // fraction

client_train_loaders = []
print(f"\n==> Generating {n_models} unique datasets for model pool training...")
for i in range(n_models):
    n_model_classes = random.randint(2, 5)
    selected_classes = random.sample(range(n_classes), n_model_classes)
    train_indices = sample_indices(selected_classes, train_class_indices, samples_per_client)
    train_subset = Subset(trainset, train_indices)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    client_train_loaders.append(train_loader)
    print(f"  Model {i+1} will be trained on classes {selected_classes} with {len(train_indices)} samples.")

client_test_loaders = []
client_validation_loaders = []
client_classes = []
print(f"\n==> Generating {n_clients} unique datasets for client evaluation...")
for i in range(n_clients):
    n_client_classes = random.randint(2, 5)
    selected_classes = random.sample(range(n_classes), n_client_classes)
    client_classes.append(selected_classes)
    all_test_indices = sample_indices(selected_classes, test_class_indices, samples_per_client // 4)
    
    random.shuffle(all_test_indices)
    
    split_point = len(all_test_indices) // 2
    validation_indices = all_test_indices[:split_point]
    final_test_indices = all_test_indices[split_point:]

    validation_subset = Subset(testset, validation_indices)
    validation_loader = DataLoader(validation_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    client_validation_loaders.append(validation_loader)

    test_subset = Subset(testset, final_test_indices)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    client_test_loaders.append(test_loader)
    
    print(f"  Client {i+1} has data for classes {selected_classes}: "
          f"{len(validation_indices)} val samples, {len(final_test_indices)} test samples.")

ensemble = [VGG('VGG19').to(device) for _ in range(n_models)]
ensemble = train_ensembles_w_local_data(ensemble, client_train_loaders, device, n_epochs, 1.0, criterion, lr)

if ensemble_selection_size > n_models:
    raise ValueError(f"ensemble_selection_size ({ensemble_selection_size}) cannot be larger than n_models ({n_models})")

def evaluate_selected_ensemble(selected_ensemble, test_loader, device, criterion):
    for model in selected_ensemble: model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            all_outputs = [model(inputs) for model in selected_ensemble]
            avg_outputs = torch.mean(torch.stack(all_outputs), dim=0)
            _, predicted = avg_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total 

def evaluate_single_model_accuracy(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def select_accuracy_only_models(model_pool, num_to_select, client_test_loader, device):
    model_accuracies = []
    for model in model_pool:
        acc = evaluate_single_model_accuracy(model, client_test_loader, device)
        model_accuracies.append((acc, model))
    model_accuracies.sort(key=lambda x: x[0], reverse=True)
    return [model for acc, model in model_accuracies[:num_to_select]]

def calculate_local_risk(model, data_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0

def calculate_disagreement(model_f, model_g, ood_loader, device):
    model_f.eval()
    model_g.eval()
    total_disagreement = 0.0
    num_batches = 0
    with torch.no_grad():
        for inputs, _ in ood_loader:
            inputs = inputs.to(device)
            #симметричная KL-дивергенция
            p_f = F.softmax(model_f(inputs), dim=1)
            p_g = F.softmax(model_g(inputs), dim=1)
            p_f = p_f.clamp(min=1e-7)
            p_g = p_g.clamp(min=1e-7)
            kl_div_fg = F.kl_div(p_g.log(), p_f, reduction='batchmean')
            kl_div_gf = F.kl_div(p_f.log(), p_g, reduction='batchmean')
            disagreement = 0.5 * (kl_div_fg + kl_div_gf)

            total_disagreement += disagreement.item()
            num_batches += 1
    return total_disagreement / num_batches if num_batches > 0 else 0

def select_uncertainty_aware_models(model_pool, num_to_select, client_test_loader, ood_loader, lambda_val, device, criterion):
    selected_models = []
    candidate_models = list(model_pool)
    model_risks = {model_idx: calculate_local_risk(model, client_test_loader, device, criterion) 
                   for model_idx, model in enumerate(candidate_models)}

    for step in range(num_to_select):
        best_candidate = None
        min_score = float('inf')
        if not selected_models:
            best_candidate_idx = min(model_risks, key=model_risks.get)
            best_candidate = candidate_models[best_candidate_idx]
        else:
            for k_idx, k in enumerate(candidate_models):
                if k in selected_models: continue
                total_disagreement = 0
                for f in selected_models:
                    # Для ускорения можно кэшировать результаты, но пока так
                    total_disagreement += calculate_disagreement(f, k, ood_loader, device)           
                avg_disagreement = total_disagreement / len(selected_models)
                risk = model_risks[k_idx]
                score = risk - lambda_val * avg_disagreement
                
                if score < min_score:
                    min_score = score
                    best_candidate = k
        
        if best_candidate:
            selected_models.append(best_candidate)
            candidate_models.remove(best_candidate)

    return selected_models


print("СРАВНЕНИЕ СТРАТЕГИЙ ВЫБОРА МОДЕЛЕЙ")
print(f"Пул моделей (`ensemble`) состоит из {n_models} моделей.")
print(f"Для каждого из {n_clients} клиентов будет выбран ансамбль из {ensemble_selection_size} моделей.")
print(f"Гиперпараметр Lambda для Uncertainty-Aware: {LAMBDA}")

for i in range(n_clients):
    print(f"\n[Клиент {i+1}] (Классы: {client_classes[i]})")
    client_val_loader = client_validation_loaders[i]
    client_final_test_loader = client_test_loaders[i]

    all_class_indices = list(range(n_classes))
    ood_classes = [c for c in all_class_indices if c not in client_classes[i]]
    ood_indices = sample_indices(ood_classes, test_class_indices, samples_per_client // 4)
    ood_subset = Subset(testset, ood_indices)
    ood_loader = DataLoader(ood_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print("  --- Стратегия: Случайный выбор (Random) ---")
    selected_ensemble_random = random.sample(ensemble, ensemble_selection_size)
    accuracy_random = evaluate_selected_ensemble(selected_ensemble_random, client_final_test_loader, device, criterion)
    print(f"  -> Точность ансамбля: {accuracy_random:.4f}%")
    
    print("  --- Стратегия: Выбор по точности (Accuracy-only) ---")
    selected_ensemble_acc = select_accuracy_only_models(ensemble, ensemble_selection_size, client_val_loader, device)
    accuracy_acc = evaluate_selected_ensemble(selected_ensemble_acc, client_final_test_loader, device, criterion)
    print(f"  -> Точность ансамбля: {accuracy_acc:.4f}%")

    print(f"  --- Стратегия: Uncertainty-Aware (lambda={LAMBDA}) ---")
    selected_ensemble_unc = select_uncertainty_aware_models(ensemble, ensemble_selection_size, client_val_loader, ood_loader, LAMBDA, device, criterion)
    accuracy_unc = evaluate_selected_ensemble(selected_ensemble_unc, client_final_test_loader, device, criterion)
    print(f"  -> Точность ансамбля: {accuracy_unc:.4f}%")
