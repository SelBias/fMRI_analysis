
import torch
import torch.nn as nn

class proposed_loss(nn.Module) : 
    def __init__(self, DEVICE, K = 3, weight = None, p_star = None, eps = 1e-8) : 
        super(proposed_loss, self).__init__()
        self.DEVICE = DEVICE

        self.K = K
        self.eps = 1e-8

        if weight is None : 
            self.weight = torch.ones(K).to(DEVICE)
        else : 
            self.weight = weight.to(DEVICE)

        if p_star is None : 
            self.p_star = torch.eye(K).to(DEVICE)
        else : 
            self.p_star = p_star.to(DEVICE)


    def forward(self, prediction, target) : 
        return -torch.mean(
            self.weight[target] * torch.log(
                torch.bmm(
                    self.p_star[target].unsqueeze(1), prediction.unsqueeze(2)
                ).view(-1) + self.eps
            )
        )
    