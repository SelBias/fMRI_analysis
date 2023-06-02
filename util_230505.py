
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

# import pandas as pd
# import seaborn as sns
# from sklearn.metrics import roc_curve


def make_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MYTensorDataset(torch.utils.data.Dataset) :
    def __init__(self, *tensors) -> None:
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


def make_masking(N, train_N, SEED = 1000) : 
    np.random.seed(SEED)
    indice = np.arange(0,N)
    mask = np.zeros(N,dtype=bool)
    rand_indice = np.random.choice(N, train_N, replace = False)
    mask[rand_indice] = True
    return indice[mask], indice[~mask]


def make_arbitrary_masking(N, ind) : 
    indice = np.arange(0,N)
    mask = np.zeros(N, dtype=bool)
    mask[ind] = True
    return indice[~mask], indice[mask]



def k_fold_index(N = 150, k = 10, randomize = True, SEED = 10) : 
    indice = np.arange(0,N)
    if randomize is True : 
        np.random.seed(SEED)
        np.random.shuffle(indice)
    result = []
    for fold in np.split(indice, k) : 
        result.append(make_arbitrary_masking(N, fold))
    return result

def seq_k_fold_index(N_data = 1574, N_subj = 150, cum_index = None, k = 10, SEED = None) : 
    indice_data = np.arange(0, N_data)
    indice_subj = np.arange(0, N_subj)

    if SEED is not None : 
        np.random.seed(SEED)
        np.random.shuffle(indice_subj)

    result = []
    for fold in np.split(indice_subj, k) : 
        mask = np.zeros_like(indice_data, dtype = bool)
        for ind in fold : 
            mask[cum_index[ind]:cum_index[ind+1]] = True
        result.append([indice_data[~mask], indice_data[mask]])
    return result


def make_decision(prediction, K = 3, order = None, threshold = None) : 
    decision = torch.zeros(prediction.shape[0])
    if order is None : 
        decision = prediction.argmax(dim = 1)
    else : 
        if threshold is None : 
            threshold = [1.0/K for i in range(K)]
        for ind in order : 
            decision[prediction[:,ind] > threshold[ind]] = ind
    return decision

def make_binary_decision(prediction, threshold = 0.5) : 
    decision = torch.ones(prediction.shape[0])
    decision[prediction < threshold] = 0
    return decision

# def make_ordinal_decision(prediction, K = 3, threshold = None) : 
#     decision = torch.zeros(prediction.shape[0])
#     if threshold is None : 
#         threshold = [(i+1.0)/K for i in range(K-1)]
#     for i in range(K-1) : 
#         decision[torch.sum(prediction[:,0:(i+1)], dim = 1) < threshold[i]] = i + 1
#     return decision

def contingency_table(decision, target, K = 3) : 
    output = torch.zeros([K,K])
    for i in range(K) :
        for j in range(K) : 
            output[i][j] = torch.sum(torch.logical_and(decision == j, target == i))
    return output


def multiclass_accuracy(decision, target, K = 3) : 
        return (decision == target).float().mean().item()

def binary_accuracy(prediction, target, threshold = 0.5) : 
    return ((prediction - threshold) * (target * 2 - 1) > 0).float().mean().item()

