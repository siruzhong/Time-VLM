import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
from sklearn.decomposition import PCA

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        print(f"The current model save path is: {path + '/' + 'checkpoint.pth'}")  # Add this line to print the save path
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def load_content(args):
    dataset_name = os.path.basename(os.path.normpath(args.root_path))
    print(dataset_name)
    if 'ETT' in args.data:
        file = 'ETT'
    elif dataset_name == 'traffic':
        file = 'Traffic'
    elif dataset_name == 'electricity':
        file = 'ECL'
    elif dataset_name == 'weather':
        file = 'Weather'
    elif dataset_name == 'illness':
        file = 'ILI'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content


def visualize_embeddings_difference(patch_features, fused_features, save_path='embedding_difference.png'):
    """
    Visualize the difference between patch_features and fused_features.
    """
    fused_mean, fused_var = fused_features.mean(), fused_features.var()
    patch_mean, patch_var = patch_features.mean(), patch_features.var()
    print(f"Fused Features - Mean: {fused_mean}, Variance: {fused_var}")
    print(f"Patch Features - Mean: {patch_mean}, Variance: {patch_var}")
    cosine_sim = torch.nn.functional.cosine_similarity(fused_features, patch_features, dim=-1)
    print(f"Cosine Similarity: {cosine_sim.mean()}")
                

def visualize_embeddings(patch_features, fused_features, save_path='embedding_distribution.png'):
    """
    Visualize the spatial distribution of patch_embedding and fused_embedding.
    """
    # Ensure inputs are PyTorch tensors
    if not isinstance(patch_features, torch.Tensor):
        patch_features = torch.tensor(patch_features)
    if not isinstance(fused_features, torch.Tensor):
        fused_features = torch.tensor(fused_features)

    patch_embedding = patch_features.reshape(-1, patch_features.size(-1))  # [B * pred_len, n_vars]
    fused_embedding = fused_features.reshape(-1, fused_features.size(-1))  # [B * 16, hidden_size]
    
    # Move tensors from GPU to CPU and convert to NumPy arrays
    patch_embedding = patch_embedding.detach().cpu().numpy()
    fused_embedding = fused_embedding.detach().cpu().numpy()

    # Randomly sample 1000 points
    num_samples = 1000
    patch_embedding = patch_embedding[np.random.choice(patch_embedding.shape[0], num_samples, replace=False)]
    fused_embedding = fused_embedding[np.random.choice(fused_embedding.shape[0], num_samples, replace=False)]

    # Reduce dimensions using UMAP
    umap = UMAP(n_components=2, random_state=42)
    patch_embedding_2d = umap.fit_transform(patch_embedding)
    fused_embedding_2d = umap.fit_transform(fused_embedding)

    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(patch_embedding_2d[:, 0], patch_embedding_2d[:, 1], c='blue', label='Patch Embedding', alpha=0.6)
    plt.scatter(fused_embedding_2d[:, 0], fused_embedding_2d[:, 1], c='red', label='Fused Embedding', alpha=0.6)
    plt.legend()
    plt.title('Patch Embedding vs Fused Embedding (2D UMAP)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(save_path)
    plt.close()

def visualize_gate_weights(gate_weights, save_path='gate_weights_distribution.png'):
    """
    Visualize the distribution of gate weights.

    Args:
        gate_weights (torch.Tensor): Gate weights with shape [B, pred_len, 2].
        save_path (str): Path to save visualization, defaults to 'gate_weights_distribution.png'.
    """
    # Extract weights for fused_features and patch_features
    fused_weights = gate_weights[:, :, 0].detach().cpu().numpy().flatten()  # Extract fused_features weights
    patch_weights = gate_weights[:, :, 1].detach().cpu().numpy().flatten()  # Extract patch_features weights

    # Create visualization
    plt.figure(figsize=(10, 5))
    plt.hist(fused_weights, bins=50, alpha=0.5, label='Fused Features Weights')
    plt.hist(patch_weights, bins=50, alpha=0.5, label='Patch Features Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Gate Weights')
    plt.legend()
    plt.savefig(save_path)
    plt.close()