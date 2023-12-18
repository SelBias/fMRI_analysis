
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
In this python file, we define severl neural networks. 

Attention(h_dim) : attention layer. This architecture is applied on both age prediction task and classfication task. 

binary_hybrid : neural networks for binary classification
 - pretrain(fMRI, mmse, age) : Pre-train using fMRI, MMSE, age
    input :  torch tensor [N x 93 x 5937] fMRI 
             torch tensor [N]             MMSE score
             torch tensor [N]             age
    output : torch tensor [N]             (estimated) probability for being non-CN (patient)
 - forwad(fMRI, mmse, age, age_diff) : After pre-training, improve the classification using fMRI, MMSE, age, age_diff
    input :  torch tensor [N x 93 x 5937] fMRI 
             torch tensor [N]             MMSE score
             torch tensor [N]             age
             torch tensor [N]             age difference
    output : torch tensor [N]             (estimated) probability for being non-CN (patient)

multi_hybrid : neural networks for multi-classification, e.g., ternary or quaternary
 - pretrain(fMRI, mmse, age) : Pre-train using fMRI, MMSE, age
    input :  torch tensor [N x 93 x 5937] fMRI 
             torch tensor [N]             MMSE score
             torch tensor [N]             age
    output : torch tensor [N x K]         (estimated) probability for each class
 - forwad(fMRI, mmse, age, age_diff) : After pre-training, improve the classification using fMRI, MMSE, age, age_diff
    input :  torch tensor [N x 93 x 5937] fMRI 
             torch tensor [N]             MMSE score
             torch tensor [N]             age
             torch tensor [N]             age difference
    output : torch tensor [N x K]         (estimated) probability for each class
'''

class Attention(nn.Module) : 
    def __init__(self, h_dim) : 
        super(Attention, self).__init__()

        self.h_dim = h_dim
        
        std = 1.0 / np.sqrt(self.h_dim)

        self.W = nn.Parameter(torch.zeros([self.h_dim, self.h_dim]))
        self.b = nn.Parameter(torch.zeros([1, 1, self.h_dim]))
        self.u = nn.Parameter(torch.zeros([self.h_dim, 1]))
        
        # Initialization
        nn.init.uniform_(self.W, -std, std)
        nn.init.uniform_(self.b, -std, std)
        nn.init.uniform_(self.u, -std, std)
        
    def forward(self, input) : 
        # input : torch array [N x L x H] (hidden state sequence)

        batch_size = input.shape[0]
        output = torch.tanh(torch.bmm(input, self.W.repeat(batch_size, 1, 1)) + self.b)  # [N x L x H] -> [N x L x H]
        output = torch.bmm(output, self.u.repeat(batch_size, 1, 1)).squeeze(2)           # [N x L x H] -> [N x L]
        return torch.softmax(output, dim = 1)                   # [N x L]


class age_estimator(nn.Module) : 
    def __init__(self, h_dim = None, num_layers = None) : 
        super(age_estimator, self).__init__()

        self.h_dim = h_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size = 5937, hidden_size = self.h_dim, num_layers = self.num_layers, 
                            batch_first = True)
        self.attention = Attention(self.h_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.h_dim + 1, 2 * self.h_dim), 
            nn.LeakyReLU(), 
            nn.Dropout(p = 0.3), 
            # nn.Linear(2 * self.h_dim, 2 * self.h_dim), 
            # nn.LeakyReLU(), 
            nn.Linear(2 * self.h_dim, self.h_dim), 
            nn.LeakyReLU(), 
            nn.Dropout(p = 0.3), 
            nn.Linear(self.h_dim, 1)
        )

    def forward(self, input, input_mmse) : 
        output, (_, _) = self.lstm(input)                       # output : [ N x L x H ]
        attn_weight = self.attention(output).unsqueeze(1)       # weight : [ N x 1 x L ]
        output = torch.bmm(attn_weight, output).squeeze(1)      # output : [ N x L ]
        output = torch.cat([
                output, 
                input_mmse.unsqueeze(1) / 30.0
            ], dim = 1)                     # output : [ N x (L+1) ]
        output = self.fc(output).squeeze(1) # output : [ N ]

        return output




class binary_hybrid(nn.Module) : 
    def __init__(self, h_dim = None, num_layers = None, K = 2) : 
        super(binary_hybrid, self).__init__()

        self.h_dim = h_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size = 5937, hidden_size = self.h_dim, 
                            num_layers = self.num_layers, batch_first = True)

        self.attention = Attention(self.h_dim)

        self.fc1 = nn.Sequential(
            nn.Linear(self.h_dim, 2 * self.h_dim), 
            nn.LeakyReLU(), 
            nn.Dropout(p=0.3), 
            nn.Linear(2 * self.h_dim, self.h_dim), 
            nn.LeakyReLU(), 
            nn.Dropout(p=0.3), 
            nn.Linear(self.h_dim + 2, 1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1, 2 * self.h_dim), 
            nn.LeakyReLU(), 
            nn.Dropout(p=0.3), 
            nn.Linear(2 * self.h_dim, self.h_dim), 
            nn.LeakyReLU(), 
            nn.Dropout(p=0.3), 
            nn.Linear(self.h_dim + 2, 1)
        )
        
    def pretrain(self, input, mmse = None, age = None, age_diff = None) : 
        output, (_, _) = self.lstm(input)                    # output : [ N x L x H ]
        attn_weight = self.attention(output).unsqueeze(1)    # weight : [ N x 1 x L ]

        output = torch.bmm(attn_weight, output).squeeze(1)   # output : [ N x H ]
        output = self.fc1(output)                            # output : [ N x H ]
        output = torch.cat([
            output, 
            mmse.unsqueeze(1) / 30.0, 
            age.unsqueeze(1) / 80.0
        ], dim = 1)                                           # output : [ N x (H+2)]
        output = self.fc2(output).flatten()                   # output : [ N ]
        
        return F.sigmoid(output)

    def forward(self, input, mmse = None, age = None, age_diff = None) : 
        output, (_, _) = self.lstm(input)                    # output : [ N x L x H ]
        attn_weight = self.attention(output).unsqueeze(1)    # weight : [ N x 1 x L ]

        output = torch.bmm(attn_weight, output).squeeze(1)   # output : [ N x H ]
        output = self.fc1(output)                            # output : [ N x H ]
        output = torch.cat([
            output, 
            mmse.unsqueeze(1) / 30.0, 
            age.unsqueeze(1) / 80.0
        ], dim = 1)                                           # output : [ N x (H+2)]
        output = self.fc2(output).flatten()                   # output : [ N ]
        output = output + self.fc3(age_diff.unsqueeze(1)).flatten()
        
        return F.sigmoid(output)


class multi_hybrid(nn.Module) : 
    def __init__(self, h_dim = None, num_layers = None, K = 4) : 
        super(multi_hybrid, self).__init__()
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.K = K
        self.lstm = nn.LSTM(input_size = 5937, hidden_size = self.h_dim, 
                            num_layers = self.num_layers, batch_first = True)
        self.attention = Attention(self.h_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(self.h_dim + 2, 2 * self.h_dim), 
            nn.LeakyReLU(), 
            nn.Dropout(p=0.3), 
            nn.Linear(2 * self.h_dim, self.h_dim), 
            nn.LeakyReLU(), 
            nn.Dropout(p=0.3), 
            nn.Linear(self.h_dim, K)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1, h_dim), 
            nn.LeakyReLU(), 
            nn.Dropout(p=0.3), 
            nn.Linear(h_dim, h_dim), 
            nn.LeakyReLU(), 
            nn.Dropout(p=0.3), 
            nn.Linear(h_dim, K, bias = False)
        )

    def pre_forward(self, input, mmse = None, age = None, age_diff = None) : 
        output, (_, _) = self.lstm(input)                   # output : [ N x L x H ]
        attn_weight = self.attention(output).unsqueeze(1)   # weight : [ N x 1 x L ]
        output = torch.bmm(attn_weight, output).squeeze(1)  # output : [ N x H ]
        output = torch.cat([
            output, 
            mmse.unsqueeze(1) / 30.0, 
            age.unsqueeze(1) / 80.0
        ], dim = 1)                                         # output : [ N x (H+2) ]
        output = self.fc1(output)                  
        return F.softmax(output)

    def forward(self, input, mmse = None, age = None, age_diff = None) : 
        output, (_, _) = self.lstm(input)                   # output : [ N x L x H ]
        attn_weight = self.attention(output).unsqueeze(1)   # weight : [ N x 1 x L ]
        output = torch.bmm(attn_weight, output).squeeze(1)  # output : [ N x H ]
        output = torch.cat([
            output, 
            mmse.unsqueeze(1) / 30.0, 
            age.unsqueeze(1) / 80.0
        ], dim = 1)                                         # output : [ N x (H+2) ]
        output = self.fc1(output) + self.fc2(age_diff.unsqueeze(1))        
        return F.softmax(output)

################################################
#####         age estimator                #####
################################################


