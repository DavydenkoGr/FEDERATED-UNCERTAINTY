import sys
import argparse
from pathlib import Path
import datetime

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
from federated_uncertainty.optim.train import train_ensembles_w_local_data
from federated_uncertainty.randomness import set_all_seeds
from federated_uncertainty.models import *
from federated_uncertainty.eval import evaluate_single_model_accuracy, evaluate_selected_ensemble

seed = 1
set_all_seeds(seed)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--n_models', default=20, type=int, help='number of models')
parser.add_argument('--n_clients', default=5, type=int, help='number of clients')
parser.add_argument('--ensemble_selection_size', default=3, type=int, help='number of models for single client')
parser.add_argument('--lambda_disagreement', default=0.1, type=float, help='disagreement importance')
parser.add_argument('--lambda_antireg', default=0.01, type=float, help='antiregularization coefficient')
parser.add_argument('--fraction', default=0.25, type=float, help='client and model part of train data')
parser.add_argument('--n_epochs', default=5, type=int, help='number of training epoches')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for models')
parser.add_argument('--model_pool_split_ratio', default=0.6, type=float, help='model/client data')
parser.add_argument('--model_min_classes', default=5, type=int, help='min classes for model pool')
parser.add_argument('--model_max_classes', default=8, type=int, help='max classes for model pool')
parser.add_argument('--client_min_classes', default=2, type=int, help='min classes for clients')
parser.add_argument('--client_max_classes', default=5, type=int, help='max classes for clients')
parser.add_argument('--save_dir', default=f'./data/saved_models/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}', type=str, help='Path to save/load ensemble models')
args = parser.parse_args()

# USE CUDA PREFFERED
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# command line arguments
n_models = args.n_models
n_clients = args.n_clients
ensemble_selection_size = args.ensemble_selection_size
lambda_disagreement = args.lambda_disagreement
lambda_antireg = args.lambda_antireg
fraction = args.fraction
n_epochs = args.n_epochs
lr = args.lr
batch_size = args.batch_size
model_pool_split_ratio = args.model_pool_split_ratio
model_min_classes = args.model_min_classes
model_max_classes = args.model_max_classes
client_min_classes = args.client_min_classes
client_max_classes = args.client_max_classes
save_dir = args.save_dir

# FOR CIFAR10 ONLY
n_classes = 10
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
all_class_indices = list(range(n_classes))
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

print('==> Splitting trainset into model_pool and client_data (stratified)...')

model_pool_class_indices = {i: [] for i in range(n_classes)}
client_data_class_indices = {i: [] for i in range(n_classes)}

total_model_pool_indices_count = 0
total_client_data_indices_count = 0

for c in range(n_classes):
    class_idx_list = train_class_indices[c]
    random.shuffle(class_idx_list)
    
    split_point = int(len(class_idx_list) * model_pool_split_ratio)
    
    model_pool_class_indices[c] = class_idx_list[:split_point]
    client_data_class_indices[c] = class_idx_list[split_point:]
    
    total_model_pool_indices_count += len(model_pool_class_indices[c])
    total_client_data_indices_count += len(client_data_class_indices[c])

print(f"  Total train data: {len(trainset)}")
print(f"  Model pool data size: {total_model_pool_indices_count} ({(total_model_pool_indices_count/len(trainset)*100):.2f}%)")
print(f"  Client data size: {total_client_data_indices_count} ({(total_client_data_indices_count/len(trainset)*100):.2f}%)")

def sample_indices(selected_classes, class_indices, total_samples):
    per_class = total_samples // len(selected_classes) if len(selected_classes) > 0 else 0
    selected_indices = []
    for c in selected_classes:
        n = min(per_class, len(class_indices[c]))
        selected_indices += random.sample(class_indices[c], n)
    return selected_indices

samples_per_model = int(len(trainset) * fraction)
samples_per_client = int(len(trainset) * fraction)

print(f"\n==> Generating {n_models} datasets for model pool training...")

model_train_loaders = []

for i in range(n_models):
    n_model_classes = random.randint(model_min_classes, model_max_classes)
    ind_classes = random.sample(range(n_classes), n_model_classes)
    train_indices = sample_indices(ind_classes, model_pool_class_indices, samples_per_model)
    train_subset = Subset(trainset, train_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    model_train_loaders.append(train_loader)
    print(f"  Model {i + 1} will be trained on classes {ind_classes} with {len(train_indices)} samples.")

print(f"\n==> Generating {n_clients} datasets for client evaluation...")

client_classes = []

client_ind_train_loaders = []
client_ood_test_loaders = []
client_ind_test_loaders = []

for i in range(n_clients):
    n_client_classes = random.randint(client_min_classes, client_max_classes)
    ind_classes = random.sample(range(n_classes), n_client_classes)
    client_classes.append(ind_classes)

    # client_ind_train_loaders
    train_ind_indices = sample_indices(ind_classes, client_data_class_indices, samples_per_client)
    train_ind_subset = Subset(trainset, train_ind_indices)
    train_ind_loader = DataLoader(train_ind_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    client_ind_train_loaders.append(train_ind_loader)

    ood_classes = [c for c in all_class_indices if c not in ind_classes]

    # client_ood_test_loaders
    test_ood_indices = sample_indices(ood_classes, test_class_indices, len(testset))

    test_ood_test_subset = Subset(testset, test_ood_indices)
    test_ood_test_loader = DataLoader(test_ood_test_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    client_ood_test_loaders.append(test_ood_test_loader)

    # client_ind_test_loaders
    test_ind_indices = sample_indices(ind_classes, test_class_indices, len(testset))

    test_ind_test_subset = Subset(testset, test_ind_indices)
    test_ind_test_loader = DataLoader(test_ind_test_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    client_ind_test_loaders.append(test_ind_test_loader)
    
    print(f"  Client {i + 1} has data for classes {ind_classes}: "
          f"{len(train_ind_indices)} train samples, {len(test_ood_indices)} ood test samples, {len(test_ind_indices)} ind test samples.")

print('\n==> Preparing ensemble path and checking for existing models...')

run_dir = Path(save_dir)
model_file_path = run_dir / 'ensemble.pt'

model_file_path.parent.mkdir(parents=True, exist_ok=True)

ensemble = [VGG('VGG19').to(device) for _ in range(n_models)]
ensemble_state_dicts = None

if model_file_path.exists():
    try:
        ensemble_state_dicts = torch.load(model_file_path)
        print(f"  Ensemble loaded from {model_file_path}.")
    except Exception as e:
        print(f"  Warning: Could not load models from {model_file_path}. Error: {e}")
        ensemble_state_dicts = None

if ensemble_state_dicts is not None:
    if len(ensemble_state_dicts) == n_models:
        for model, state_dict in zip(ensemble, ensemble_state_dicts):
            model.load_state_dict(state_dict)
    else:
        print(f"  Warning: Expected {n_models} models, but loaded {len(ensemble_state_dicts)}. Retraining.")
        ensemble_state_dicts = None

if ensemble_state_dicts is None:
    print("  Training ensemble from scratch...")
    
    ensemble = train_ensembles_w_local_data(
        ensemble, 
        model_train_loaders, 
        device, 
        n_epochs, 
        lambda_antireg,
        criterion,
        lr
    )

    state_dicts_to_save = [model.state_dict() for model in ensemble]
    torch.save(state_dicts_to_save, model_file_path)
    print(f"  Ensemble successfully trained and saved to {model_file_path}.")

if ensemble_selection_size > n_models:
    raise ValueError(f"ensemble_selection_size ({ensemble_selection_size}) cannot be larger than n_models ({n_models})")

def select_accuracy_only_models(model_pool, num_to_select, client_test_loader, device):
    accuracies = []

    for i, model in enumerate(model_pool):
        acc = evaluate_single_model_accuracy(model, client_test_loader, device)
        accuracies.append((acc, i))

    accuracies.sort(key=lambda x: x[0], reverse=True)

    return [idx for acc, idx in accuracies[:num_to_select]]

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
            p_f = F.softmax(model_f(inputs), dim=1)
            p_g = F.softmax(model_g(inputs), dim=1)
            p_f = p_f.clamp(min=1e-7)
            p_g = p_g.clamp(min=1e-7)

            disagreement = F.kl_div(p_f.log(), p_g, reduction='batchmean')

            total_disagreement += disagreement.item()
            num_batches += 1
    return total_disagreement / num_batches if num_batches > 0 else 0

def select_uncertainty_aware_models(
        model_pool,
        num_to_select,
        client_test_loader,
        ood_loader,
        lambda_val,
        device,
        criterion):

    candidate_indices = list(range(len(model_pool)))

    model_risks = {
        idx: calculate_local_risk(model_pool[idx], client_test_loader, device, criterion)
        for idx in candidate_indices
    }

    selected_indices = []

    for step in range(num_to_select):
        best_idx = None
        best_score = float('inf')

        if not selected_indices:
            best_idx = min(model_risks, key=model_risks.get)
        else:
            for k_idx in candidate_indices:
                if k_idx in selected_indices:
                    continue

                total_disagreement = 0
                for f_idx in selected_indices:
                    total_disagreement += calculate_disagreement(
                        model_pool[f_idx],
                        model_pool[k_idx],
                        ood_loader,
                        device
                    )

                avg_disagreement = total_disagreement / len(selected_indices)
                risk = model_risks[k_idx]
                score = risk - lambda_val * avg_disagreement

                if score < best_score:
                    best_score = score
                    best_idx = k_idx

        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)

    return selected_indices

print("MODEL SELECTION STRATEGIES COMPARISON")
print(f"The model pool (`ensemble`) consists of {n_models} models.")
print(f"For each of the {n_clients} clients, an ensemble of {ensemble_selection_size} models will be selected.")
print(f"Lambda hyperparameter for Uncertainty-Aware: {lambda_disagreement}")

for i in range(n_clients):
    print(f"\n[Client {i + 1}] (Classes: {client_classes[i]})")

    client_ind_train_loader = client_ind_train_loaders[i]
    client_ood_test_loader = client_ood_test_loaders[i]
    client_ind_test_loader = client_ind_test_loaders[i]

    # Random Selection
    print("  --- Strategy: Random Selection ---")
    ensemble_random_indices = random.sample(range(n_models), ensemble_selection_size)
    selected_ensemble_random = [ensemble[i] for i in ensemble_random_indices]
    print(f"  -> Selected models: {ensemble_random_indices}")

    accuracy_random = evaluate_selected_ensemble(selected_ensemble_random, client_ind_test_loader, device, criterion)
    for i, model in enumerate(selected_ensemble_random):
        accuracy_model = evaluate_single_model_accuracy(model, client_ind_test_loader, device)
        print(f"  -> Model[{i + 1}] accuracy: {accuracy_model:.4f}%")
    print(f"  -> Ensemble accuracy: {accuracy_random:.4f}%")
    
    # Accuracy-only Selection
    print("\n  --- Strategy: Accuracy-only Selection ---")
    ensemble_acc_indices = select_accuracy_only_models(ensemble, ensemble_selection_size, client_ind_train_loader, device)
    selected_ensemble_acc = [ensemble[i] for i in ensemble_acc_indices]
    print(f"  -> Selected models: {ensemble_acc_indices}")

    accuracy_acc = evaluate_selected_ensemble(selected_ensemble_acc, client_ind_test_loader, device, criterion)
    for i, model in enumerate(selected_ensemble_acc):
        accuracy_model = evaluate_single_model_accuracy(model, client_ind_test_loader, device)
        print(f"  -> Model[{i + 1}] accuracy: {accuracy_model:.4f}%")
    print(f"    -> Ensemble accuracy: {accuracy_acc:.4f}%")

    # Uncertainty-Aware
    print(f"\n  --- Strategy: Uncertainty-Aware (lambda={lambda_disagreement}) ---")
    ensemble_unc_indices = select_uncertainty_aware_models(ensemble, ensemble_selection_size, client_ind_train_loader, client_ood_test_loader, lambda_disagreement, device, criterion)
    selected_ensemble_unc = [ensemble[i] for i in ensemble_unc_indices]
    print(f"  -> Selected models: {ensemble_unc_indices}")

    accuracy_unc = evaluate_selected_ensemble(selected_ensemble_unc, client_ind_test_loader, device, criterion)
    for i, model in enumerate(selected_ensemble_unc):
        accuracy_model = evaluate_single_model_accuracy(model, client_ind_test_loader, device)
        print(f"  -> Model[{i + 1}] accuracy: {accuracy_model:.4f}%")
    print(f"    -> Ensemble accuracy: {accuracy_unc:.4f}%")
