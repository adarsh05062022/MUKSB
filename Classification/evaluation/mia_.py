"""Membership Inference Attack via per-sample loss (logistic regression)."""
import gc

import numpy as np
import torch
from sklearn import linear_model, model_selection


def compute_losses(net, loader, device):
    """Compute per-sample cross-entropy losses."""
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            logits = net(inputs)
        losses = criterion(logits, targets).detach().cpu().numpy()
        all_losses.extend(losses.tolist())
        torch.cuda.empty_cache()
        gc.collect()
    return np.array(all_losses)


def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    unique = np.unique(members)
    if not np.all(unique == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")
    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state)
    return model_selection.cross_val_score(attack_model, sample_loss, members, cv=cv, scoring="accuracy")


def get_mia(net, member_loader, nonmember_loader, device, n_splits=10, random_state=0):
    """Return mean MIA accuracy (0.5 = random, 1.0 = perfect membership inference)."""
    loss_mem = compute_losses(net, member_loader, device)
    loss_non = compute_losses(net, nonmember_loader, device)
    if len(loss_mem) > len(loss_non):
        np.random.shuffle(loss_mem); loss_mem = loss_mem[:len(loss_non)]
    else:
        np.random.shuffle(loss_non); loss_non = loss_non[:len(loss_mem)]
    samples_loss  = np.concatenate((loss_mem, loss_non)).reshape((-1, 1))
    label_members = [1] * len(loss_mem) + [0] * len(loss_non)
    return simple_mia(samples_loss, label_members, n_splits, random_state).mean()
