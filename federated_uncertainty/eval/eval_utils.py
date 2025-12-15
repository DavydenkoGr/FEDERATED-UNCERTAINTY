import torch


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
    return correct / total


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
    return correct / total