import sys
import argparse
from pathlib import Path
import datetime
import random
import copy
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated_uncertainty.optim.train import train_ensembles_w_local_data
from federated_uncertainty.randomness import set_all_seeds
from federated_uncertainty.nn import QuantVGG, VGG
from federated_uncertainty.eval import evaluate_single_model_accuracy, evaluate_selected_ensemble
from federated_uncertainty.noise import get_noisy_model, NoiseType, NOISE_CHOICES
from federated_uncertainty.data import load_dataset, get_class_indices

parser = argparse.ArgumentParser(description='PyTorch FEDERATED UNCERTAINTY Training')
parser.add_argument('--n_models', default=20, type=int, help='number of models')
parser.add_argument('--n_clients', default=5, type=int, help='number of clients')
parser.add_argument('--ensemble_size', default=3, type=int, help='number of models for single client')
parser.add_argument('--lambda_disagreement', default=0.1, type=float, help='disagreement importance')
parser.add_argument('--lambda_antireg', default=0.01, type=float, help='antiregularization coefficient')
parser.add_argument('--fraction', default=0.25, type=float, help='client and model part of train data')
parser.add_argument('--n_epochs', default=50, type=int, help='number of training epoches')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for models')
parser.add_argument('--model_pool_split_ratio', default=0.6, type=float, help='model/client data')
parser.add_argument('--model_min_classes', default=5, type=int, help='min classes for model pool')
parser.add_argument('--model_max_classes', default=8, type=int, help='max classes for model pool')
parser.add_argument('--client_min_classes', default=2, type=int, help='min classes for clients')
parser.add_argument('--client_max_classes', default=5, type=int, help='max classes for clients')
parser.add_argument('--noise_type', 
                    default=NoiseType.QUANT.name.lower(),
                    type=str, 
                    choices=NOISE_CHOICES,
                    help=f'Type of noise to apply to spoiler models. Choices: {", ".join(NOISE_CHOICES)}')
parser.add_argument('--spoiler_noise', default=0.05, type=float, help='std of noise added to spoiler weights')
parser.add_argument('--market_lr', default=1.0, type=float, help='learning rate for mirror descent')
parser.add_argument('--market_epochs', default=2, type=int, help='optimization steps for market weighting')
parser.add_argument('--save_dir', 
                    default=f"./data/saved_models/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                    type=str, help='Path to save/load ensemble models')
parser.add_argument('--dataset', default='cifar10', type=str, 
                    choices=['cifar10', 'cifar100', 'PathMNIST', 'OrganAMNIST', 'TissueMNIST', 'BloodMNIST'], 
                    help='dataset to use (cifar10, cifar100, PathMNIST, OrganAMNIST, TissueMNIST, BloodMNIST)')
parser.add_argument('--seed', default=0, type=int, help='seed for random number generator')

args = parser.parse_args()

seed = args.seed
set_all_seeds(seed)

# USE CUDA PREFFERED
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# command line arguments
n_models = args.n_models
n_clients = args.n_clients
ensemble_size = args.ensemble_size
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
spoiler_noise = args.spoiler_noise
noise_type = NoiseType[args.noise_type.upper()]
save_dir = args.save_dir
dataset_name = args.dataset

# load dataset
trainset, testset, n_classes = load_dataset(dataset_name, data_root='./data')
all_class_indices = list(range(n_classes))
criterion = nn.CrossEntropyLoss()

train_class_indices = get_class_indices(trainset, n_classes)
test_class_indices = get_class_indices(testset, n_classes)

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

print(f"\n==> Generating {n_models} datasets for model training...")

model_train_loaders = []

for i in range(n_models):
    # maybe better to add raise error
    model_max_classes = min(model_max_classes, n_classes)
    model_min_classes = min(model_min_classes, model_max_classes)

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

# Save hyperparameters to YAML config
config_path = run_dir / 'config.yaml'
config_dict = vars(args)
with open(config_path, 'w') as f:
    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
print(f"  Hyperparameters saved to {config_path}.")

ensemble = [VGG('VGG19', n_classes).to(device) for _ in range(n_models)]
ensemble_state_dicts = None

if model_file_path.exists():
    try:
        ensemble_state_dicts = torch.load(model_file_path, map_location=device)
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

if ensemble_size > n_models:
    raise ValueError(f"ensemble_size ({ensemble_size}) cannot be larger than n_models ({n_models})")

# Ablation study: spoile noise levels to test
spoiler_noise_values = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 5e-6]

print(f"\n{'='*60}")
print(f"SPOILER NOISE ABLATION STUDY")
print(f"Testing spoiler noise levels: {spoiler_noise_values}")
print(f"{'='*60}\n")

def get_all_logits_and_targets(models, data_loader, device):
    all_inputs = []
    all_targets = []
    
    for inp, tar in data_loader:
        all_inputs.append(inp)
        all_targets.append(tar)
        
    inputs_tensor = torch.cat(all_inputs, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0).to(device)
    
    n_samples = inputs_tensor.shape[0]
    all_logits = []

    with torch.no_grad():
        for model in models:
            model.eval()
            model.to(device)
            model_logits = []
            
            for i in range(0, n_samples, batch_size):
                batch_inp = inputs_tensor[i : i + batch_size].to(device)
                model_logits.append(model(batch_inp).cpu())
            
            all_logits.append(torch.cat(model_logits, dim=0))
            
    logits_tensor = torch.stack(all_logits).to(device)
    return logits_tensor, targets_tensor

def optimize_ensemble_weights(models, client_loader, device, args):
    print(f"    [Weight Opt] Optimizing weights for {len(models)} models...")
    
    logits_tensor, targets_tensor = get_all_logits_and_targets(models, client_loader, device)
    
    probs = F.softmax(logits_tensor, dim=-1)
    log_probs = F.log_softmax(logits_tensor, dim=-1)
    
    p_i = probs.unsqueeze(1)
    log_p_i = log_probs.unsqueeze(1)
    log_p_j = log_probs.unsqueeze(0)
    
    kl_pairwise_samples = torch.sum(p_i * (log_p_i - log_p_j), dim=-1) 
    disagreement_matrix = kl_pairwise_samples.mean(dim=-1).to(device)
    disagreement_matrix.fill_diagonal_(0)
    
    n_m = len(models)
    w = torch.ones(n_m, device=device) / n_m
    w = w.detach().requires_grad_(True)
    
    for _ in range(args.market_epochs):
        probs_models = F.softmax(logits_tensor, dim=-1)
        probs_ens = torch.einsum('m, mnc -> nc', w, probs_models)
        
        loss_nll = F.nll_loss(torch.log(probs_ens + 1e-9), targets_tensor)
        w_D = torch.matmul(w, disagreement_matrix) 
        diversity = torch.dot(w_D, w)
        
        loss = loss_nll - args.lambda_disagreement * diversity
        
        if w.grad is not None: w.grad.zero_()
        loss.backward()
        
        with torch.no_grad():
            w_new = w * torch.exp(-args.market_lr * w.grad)
            w_new /= w_new.sum()
            w.copy_(w_new)
            w.grad.zero_()

    weights_np = w.detach().cpu().numpy()
    print(f"    [Weight Opt] Final Weights: {weights_np}")
    
    return w.detach()

def evaluate_weighted_ensemble(models, weights, data_loader, device, criterion):
    total_loss = 0.0
    correct = 0
    total = 0
    weights = weights.to(device)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs_list = [model(inputs) for model in models]
            outputs_stack = torch.stack(outputs_list, dim=1) 
            probs_stack = F.softmax(outputs_stack, dim=-1)
            
            ensemble_probs = torch.einsum('m, bmc -> bc', weights, probs_stack)
            
            loss = criterion(torch.log(ensemble_probs + 1e-9), targets)
            total_loss += loss.item() * inputs.size(0)
            
            _, predicted = ensemble_probs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_loss = total_loss / total
    return acc, avg_loss

def select_accuracy_only_models(spoilers, num_to_select, client_test_loader, device):
    accuracies = []

    for i, model in enumerate(spoilers):
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
        spoilers,
        num_to_select,
        client_test_loader,
        ood_loader,
        lambda_disagreement,
        device,
        criterion):

    candidate_indices = list(range(len(spoilers)))

    model_risks = {
        idx: calculate_local_risk(spoilers[idx], client_test_loader, device, criterion)
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
                        spoilers[f_idx],
                        spoilers[k_idx],
                        ood_loader,
                        device
                    )

                avg_disagreement = total_disagreement / len(selected_indices)
                risk = model_risks[k_idx]
                score = risk - lambda_disagreement * avg_disagreement

                if score < best_score:
                    best_score = score
                    best_idx = k_idx

        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)

    return selected_indices

def select_market_models(spoilers, num_to_select, client_loader, device, args):
    logits_tensor, targets_tensor = get_all_logits_and_targets(spoilers, client_loader, device)
    
    print("    [Market] Pre-computing pairwise disagreement matrix...")
    probs = F.softmax(logits_tensor, dim=-1)
    log_probs = F.log_softmax(logits_tensor, dim=-1)
    
    p_i = probs.unsqueeze(1)
    log_p_i = log_probs.unsqueeze(1)
    log_p_j = log_probs.unsqueeze(0)
    
    kl_pairwise_samples = torch.sum(p_i * (log_p_i - log_p_j), dim=-1) 
    disagreement_matrix = kl_pairwise_samples.mean(dim=-1).to(device)
    disagreement_matrix.fill_diagonal_(0)
    
    n_m = len(spoilers)
    w = torch.ones(n_m, device=device) / n_m
    w = w.detach().requires_grad_(True)
    
    print(f"    [Market] Optimizing weights ({args.market_epochs} steps)...")
    
    for _ in range(args.market_epochs):
        probs_models = F.softmax(logits_tensor, dim=-1)
        probs_ens = torch.einsum('n, nmc -> mc', w, probs_models)
        loss_nll = F.nll_loss(torch.log(probs_ens + 1e-7), targets_tensor)
        w_D = torch.matmul(w, disagreement_matrix) 
        diversity = torch.dot(w_D, w)
        loss = loss_nll - args.lambda_disagreement * diversity
        
        if w.grad is not None: w.grad.zero_()
        loss.backward()
        
        with torch.no_grad():
            w_new = w * torch.exp(-args.market_lr * w.grad)
            w_new /= w_new.sum()
            w.copy_(w_new)
            w.grad.zero_()

    weights_np = w.detach().cpu().numpy()
    selected_indices = weights_np.argsort()[-num_to_select:][::-1]
    
    print(f"    [Market] Final Weights Top: {weights_np[selected_indices]}")
    
    return selected_indices.tolist()


# Logits and labels saving (keeps alignment regardless of DataLoader shuffle)
def get_logits_and_labels(models, data_loader, device):
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            # Collect logits from each model in the ensemble
            batch_logits = torch.stack([model(inputs) for model in models], dim=0)  # shape: (ensemble_size, batch, classes)
            all_logits.append(batch_logits.cpu())
            all_labels.append(labels.cpu())
    # Concatenate all batches: logits shape (ensemble_size, total_samples, classes); labels shape (total_samples,)
    return torch.cat(all_logits, dim=1), torch.cat(all_labels, dim=0)


def save_logits_and_labels(selected_models, ind_loader, ood_loader, client_num, strategy, device, run_dir):
    ind_logits, y_ind = get_logits_and_labels(selected_models, ind_loader, device)
    ood_logits, y_ood = get_logits_and_labels(selected_models, ood_loader, device)

    # Save logits and labels for later use
    logits_ind_path = run_dir / "logits" / f"client_{client_num}" / f"{strategy}_ind.pt"
    logits_ood_path = run_dir / "logits" / f"client_{client_num}" / f"{strategy}_ood.pt"
    labels_ind_path = run_dir / "labels" / f"client_{client_num}" / f"{strategy}_ind.pt"
    labels_ood_path = run_dir / "labels" / f"client_{client_num}" / f"{strategy}_ood.pt"

    logits_ind_path.parent.mkdir(parents=True, exist_ok=True)
    labels_ind_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(ind_logits, logits_ind_path)
    torch.save(ood_logits, logits_ood_path)
    torch.save(y_ind, labels_ind_path)
    torch.save(y_ood, labels_ood_path)

    print(f"Saved client {client_num} logits and labels to:  {run_dir}")

def select_and_evaluate_models(
    strategy,
    ensemble,
    spoilers,
    client_ind_train_loader,
    client_ood_test_loader,
    client_ind_test_loader,
    client_num,
    device,
    criterion,
    ensemble_size,
    run_dir,
):  
    if strategy == "random":
        print("  --- Strategy: Random Selection ---")
        ensemble_indices = random.sample(range(n_models), ensemble_size)
    elif strategy == "accuracy":
        print("\n  --- Strategy: Accuracy-only Selection ---")
        ensemble_indices = select_accuracy_only_models(
            spoilers, ensemble_size, client_ind_train_loader, device
        )
    elif strategy == "uncertainty":
        print(f"\n  --- Strategy: Uncertainty-Aware (lambda={lambda_disagreement}) ---")
        ensemble_indices = select_uncertainty_aware_models(
            spoilers,
            ensemble_size,
            client_ind_train_loader,
            client_ood_test_loader,
            lambda_disagreement,
            device,
            criterion,
        )
    elif strategy == "market":
        print(f"\n  --- Strategy: Market (selection by weights) ---")
        ensemble_indices = select_market_models(
            spoilers, ensemble_size, client_ind_train_loader, device, args
        )
    else:
        raise ValueError(f"Unknown strategy '{strategy}'.")
    
    selected_ensemble = [ensemble[i] for i in ensemble_indices]
    print(f"  -> Selected models: {ensemble_indices}")

    accuracy = evaluate_selected_ensemble(selected_ensemble, client_ind_test_loader, device, criterion)
    for i, model in enumerate(selected_ensemble):
        accuracy_model = evaluate_single_model_accuracy(model, client_ind_test_loader, device)
        print(f"  -> Model[{i + 1}] accuracy: {accuracy_model:.4f}")
    print(f"    -> Ensemble accuracy: {accuracy:.4f}")

    save_logits_and_labels(selected_ensemble, client_ind_test_loader, client_ood_test_loader, client_num, strategy, device, run_dir)

    return selected_ensemble, ensemble_indices


# Ablation loop over spoiler noise levels
for current_spoiler_noise in spoiler_noise_values:
    print(f"\n{'='*60}")
    print(f"SPOILER NOISE: {current_spoiler_noise}")
    print(f"{'='*60}")
    
    # Create separate directory for this noise level
    # Format noise value for directory name (e.g., 1e-5 -> spoiler_noise_1e-5)
    noise_str = f"{current_spoiler_noise:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    run_dir_noise = run_dir / f"spoiler_noise_{noise_str}"
    run_dir_noise.mkdir(parents=True, exist_ok=True)
    
    # Create spoiler versions of our models from pool for this noise level
    spoilers = []
    for model in ensemble:
        spoiler = get_noisy_model(model, noise_type, device, noise_level=current_spoiler_noise)
        spoilers.append(spoiler)
    
    print("MODEL SELECTION STRATEGIES COMPARISON")
    print(f"The model pool (`ensemble`) consists of {n_models} models.")
    print(f"For each of the {n_clients} clients, an ensemble of {ensemble_size} models will be selected.")
    print(f"Lambda: {lambda_disagreement}, Spoiler Noise: {current_spoiler_noise}")
    print(f"Results will be saved to: {run_dir_noise}")

    for i in range(n_clients):
        print(f"\n[Client {i + 1}] (Classes: {client_classes[i]})")

        selected_ensemble_random, indices_random = select_and_evaluate_models(
            "random",
            ensemble,
            spoilers,
            client_ind_train_loaders[i],
            client_ood_test_loaders[i],
            client_ind_test_loaders[i],
            i + 1,
            device,
            criterion,
            ensemble_size,
            run_dir_noise,
        )

        selected_ensemble_acc, indices_acc = select_and_evaluate_models(
            "accuracy",
            ensemble,
            spoilers,
            client_ind_train_loaders[i],
            client_ood_test_loaders[i],
            client_ind_test_loaders[i],
            i + 1,
            device,
            criterion,
            ensemble_size,
            run_dir_noise,
        )

        selected_ensemble_unc, indices_unc = select_and_evaluate_models(
            "uncertainty",
            ensemble,
            spoilers,
            client_ind_train_loaders[i],
            client_ood_test_loaders[i],
            client_ind_test_loaders[i],
            i + 1,
            device,
            criterion,
            ensemble_size,
            run_dir_noise,
        )

        selected_ensemble_mkt, indices_mkt = select_and_evaluate_models(
            "market",
            ensemble,
            spoilers,
            client_ind_train_loaders[i],
            client_ood_test_loaders[i],
            client_ind_test_loaders[i],
            i + 1,
            device,
            criterion,
            ensemble_size,
            run_dir_noise,
        )
        
        # print(f"\n  --- Strategy: Hybrid (Uncertainty Selection + Market Weighting) ---")
        
        # hybrid_indices = indices_unc 
        # hybrid_ensemble = selected_ensemble_unc

        # print(f"  -> Selected models: {hybrid_indices}")
        
        # hybrid_weights = optimize_ensemble_weights(
        #     hybrid_ensemble,
        #     client_ind_train_loaders[i],
        #     device,
        #     args
        # )
        
        # hybrid_acc, hybrid_loss = evaluate_weighted_ensemble(
        #     hybrid_ensemble, 
        #     hybrid_weights, 
        #     client_ind_test_loaders[i], 
        #     device, 
        #     criterion
        # )
        
        # print(f"    -> Hybrid Ensemble Accuracy (Weighted): {hybrid_acc:.4f} (Loss: {hybrid_loss:.4f})")
        
        # save_logits_and_labels(hybrid_ensemble, client_ind_test_loaders[i], client_ood_test_loaders[i], i + 1, "hybrid", device, run_dir_noise)

print(f"\n{'='*60}")
print(f"ABLATION STUDY COMPLETED")
noise_dirs = [f"spoiler_noise_{f'{v:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')}" for v in spoiler_noise_values]
print(f"Results saved in subdirectories: {noise_dirs}")
print(f"{'='*60}\n")