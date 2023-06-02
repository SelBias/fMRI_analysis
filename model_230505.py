
import numpy as np
import torch
import torch.nn as nn

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


################################################
#####     Binary classification module     #####
################################################


class binary_without_age_diff(nn.Module) : 
    def __init__(self, h_dim = None, num_layers = None, K = 2) : 
        super(binary_without_age_diff, self).__init__()

        self.h_dim = h_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size = 5937, hidden_size = self.h_dim, 
                            num_layers = self.num_layers, batch_first = True)

        self.attention = Attention(self.h_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(self.h_dim + 2, 2 * self.h_dim), 
            nn.LeakyReLU(), 
            nn.Linear(2 * self.h_dim, 2 * self.h_dim), 
            nn.LeakyReLU(), 
            nn.Dropout(), 
            nn.Linear(2 * self.h_dim, self.h_dim), 
            nn.LeakyReLU(), 
            nn.Linear(self.h_dim, 1), 
            nn.Sigmoid()
        )
        
    def forward(self, input, mmse = None, age = None, age_diff = None) : 
        output, (_, _) = self.lstm(input)                    # output : [ N x L x H ]
        attn_weight = self.attention(output).unsqueeze(1)    # weight : [ N x 1 x L ]

        output = torch.bmm(attn_weight, output).squeeze(1)   # output : [ N x H ]
        output = torch.cat([
            output, 
            mmse.unsqueeze(1) / 30.0, 
            age.unsqueeze(1) / 80.0
        ], dim = 1)                                          # output : [ N x (H+2)]
        output = self.fc(output).flatten()                   # output : [ N ]
        
        return output


class binary_with_age_diff(nn.Module) : 
    def __init__(self, h_dim = None, num_layers = None, K = 2) : 
        super(binary_with_age_diff, self).__init__()

        self.h_dim = h_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size = 5937, hidden_size = self.h_dim, 
                            num_layers = self.num_layers, batch_first = True)

        self.attention = Attention(self.h_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(self.h_dim + 3, 2 * self.h_dim), 
            nn.LeakyReLU(), 
            nn.Linear(2 * self.h_dim, 2 * self.h_dim), 
            nn.LeakyReLU(), 
            nn.Dropout(), 
            nn.Linear(2 * self.h_dim, self.h_dim), 
            nn.LeakyReLU(), 
            nn.Linear(self.h_dim, 1), 
            nn.Sigmoid()
        )
        
    def forward(self, input, mmse = None, age = None, age_diff = None) : 
        output, (_, _) = self.lstm(input)                    # output : [ N x L x H ]
        attn_weight = self.attention(output).unsqueeze(1)    # weight : [ N x 1 x L ]

        output = torch.bmm(attn_weight, output).squeeze(1)   # output : [ N x H ]
        output = torch.cat([
            output, 
            mmse.unsqueeze(1) / 30.0, 
            age.unsqueeze(1) / 80.0, 
            age_diff.unsqueeze(1) / 10.0
        ], dim = 1)                                          # output : [ N x (H+2)]
        output = self.fc(output).flatten()                   # output : [ N ]
        
        return output


################################################
#####   Multiclass classification module   #####
################################################

class multi_without_age_diff(nn.Module) : 
    def __init__(self, h_dim = None, num_layers = None, K = 4) : 
        super(multi_without_age_diff, self).__init__()
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.K = K
        self.lstm = nn.LSTM(input_size = 5937, hidden_size = self.h_dim, 
                            num_layers = self.num_layers, batch_first = True)
        self.attention = Attention(self.h_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.h_dim + 2, 2 * self.h_dim), 
            nn.LeakyReLU(), 
            nn.Linear(2 * self.h_dim, 2 * self.h_dim), 
            nn.LeakyReLU(), 
            nn.Dropout(), 
            nn.Linear(2 * self.h_dim, self.h_dim), 
            nn.LeakyReLU(), 
            nn.Linear(self.h_dim, self.K), 
            nn.Softmax(dim = 1)
        )

    def forward(self, input, mmse = None, age = None, age_diff = None) : 
        output, (_, _) = self.lstm(input)                   # output : [ N x L x H ]
        attn_weight = self.attention(output).unsqueeze(1)   # weight : [ N x 1 x L ]
        output = torch.bmm(attn_weight, output).squeeze(1)  # output : [ N x H ]
        output = torch.cat([
            output, 
            mmse.unsqueeze(1) / 30.0, 
            age.unsqueeze(1) / 80.0
        ], dim = 1)                                          # output : [ N x (H+1) ]
        output = self.fc(output)                             # output : [ N x K ]
        return output

class multi_with_age_diff(nn.Module) : 
    def __init__(self, h_dim = None, num_layers = None, K = 4) : 
        super(multi_with_age_diff, self).__init__()
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.K = K
        self.lstm = nn.LSTM(input_size = 5937, hidden_size = self.h_dim, 
                            num_layers = self.num_layers, batch_first = True)
        self.attention = Attention(self.h_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.h_dim + 3, 2 * self.h_dim), 
            nn.LeakyReLU(), 
            nn.Linear(2 * self.h_dim, 2 * self.h_dim), 
            nn.LeakyReLU(), 
            nn.Dropout(), 
            nn.Linear(2 * self.h_dim, self.h_dim), 
            nn.LeakyReLU(), 
            nn.Linear(self.h_dim, self.K), 
            nn.Softmax(dim = 1)
        )

    def forward(self, input, mmse = None, age = None, age_diff = None) : 
        output, (_, _) = self.lstm(input)                   # output : [ N x L x H ]
        attn_weight = self.attention(output).unsqueeze(1)   # weight : [ N x 1 x L ]
        output = torch.bmm(attn_weight, output).squeeze(1)  # output : [ N x H ]
        output = torch.cat([
            output, 
            mmse.unsqueeze(1) / 30.0, 
            age.unsqueeze(1) / 80.0, 
            age_diff.unsqueeze(1) / 10.0
        ], dim = 1)                                          # output : [ N x (H+2) ]
        output = self.fc(output)                             # output : [ N x K ]
        return output

################################################
#####         age estimator                #####
################################################


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
            nn.Linear(2 * self.h_dim, 2 * self.h_dim), 
            nn.LeakyReLU(), 
            nn.Dropout(), 
            nn.Linear(2 * self.h_dim, self.h_dim), 
            nn.LeakyReLU(), 
            nn.Linear(self.h_dim, 1)
        )

    def forward(self, input, input_mmse) : 
        output, (_, _) = self.lstm(input)                               # output : [ N x L x H ]
        attn_weight = self.attention(output)                                # weight : [ N x L ]
        output = torch.bmm(attn_weight.unsqueeze(1), output).squeeze(1)      # output : [ N x L ]
        output = torch.cat([
                output, 
                input_mmse.unsqueeze(1).float() / 30.0
            ], dim = 1)                     # output : [ N x (L+1) ]
        output = self.fc(output).squeeze(1) # output : [ N ]

        return output
