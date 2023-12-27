
import random
import numpy as np
import torch

def make_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TensorDataset(torch.utils.data.Dataset) :
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


def make_arbitrary_masking(N, ind, randomize = False, SEED = None) : 
    indice = np.arange(0,N)
    mask = np.zeros(N, dtype=bool)
    mask[ind] = True
    if randomize is True : 
        if SEED is not None : 
            np.random.seed(SEED)
        np.random.shuffle(mask)
    return indice[~mask], indice[mask]


def k_fold_index(N = 150, k = 10, randomize = True, SEED = None) : 
    indice = np.arange(0,N)
    if randomize is True : 
        if SEED is not None : 
            np.random.seed(SEED)
        np.random.shuffle(indice)
    result = []
    for fold in np.split(indice, k) : 
        result.append(make_arbitrary_masking(N, fold))
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


def multiclass_accuracy(decision, target, K = 3) : 
        return (decision == target).float().mean().item()


def contingency_table(decision, target, K = 3) : 
    output = torch.zeros([K,K])
    for i in range(K) :
        for j in range(K) : 
            output[i][j] = torch.sum(torch.logical_and(decision == j, target == i))
    return output


