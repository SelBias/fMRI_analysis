

import torch
import torch.nn as nn

class Binary_module(nn.Module) : 
    def __init__(self, DEVICE) : 
        super(Binary_module, self).__init__()

        self.DEVICE = DEVICE
        self.loss = nn.BCELoss().to(self.DEVICE)
    
    def forward(self, prediction, target) :
        return self.loss(prediction, target.float())

class NLL_module(nn.Module) : 
    def __init__(self, DEVICE, weight = None, eps = 1e-8) : 
        super(NLL_module, self).__init__()
        
        self.DEVICE = DEVICE
        self.loss = nn.NLLLoss(weight=weight).to(self.DEVICE)
        self.eps = eps

    def forward(self, prediction, target) : 
        return self.loss(torch.log(prediction + eps), target)

class Proposed_module(nn.Module) : 
    def __init__(self, DEVICE, K = 4, weight = None, p_star = None, eps = 1e-8) : 
        '''
        K = 3 : number of classes

        weight     : torch array [K, ]    weight for each class (default : matrix of ones)
        p          : torch array [K x K]  penalty matrix        (default : J_n - I_n)
        p_star     : torch array [K x K]  p_star = J_n - p      (default : I_n)
        '''
        super(Proposed_module, self).__init__()
        self.DEVICE = DEVICE

        self.K = K
        self.weight = weight
        self.p_star = p_star
        self.eps = 1e-8

        if self.weight is None : 
            self.weight = torch.ones(K).to(DEVICE)
        else : 
            self.weight = self.weight.to(DEVICE)

        if self.p_star is None : 
            self.p_star = torch.eye(K).to(DEVICE)
        else : 
            self.p_star = self.p_star.to(DEVICE)

    def forward(self, prediction, target) : 

        return -torch.mean(
            self.weight[target] * torch.log(
                torch.bmm(
                    self.p_star[target].unsqueeze(1), prediction.unsqueeze(2)
                ).view(-1) + self.eps
            )
        )


def loss_generator(DEVICE, loss_type = 'Proposed', K = 4, weight = None, p_star = None, eps = 1e-8) : 
    if loss_type == 'Binary' : 
        return Binary_module(DEVICE)
    elif loss_type == 'NLL' : 
        return NLL_module(DEVICE, weight, eps)
    elif loss_type == "MSE" : 
        return nn.MSELoss().to(DEVICE)
    else : 
        return Proposed_module(DEVICE, K, weight, p_star, eps)



