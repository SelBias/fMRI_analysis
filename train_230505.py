
import time
import torch
import random


import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader

# from loss_230505 import Binary_module, NLL_module, Proposed_module, loss_generator
from util_230505 import MYTensorDataset, make_masking, make_arbitrary_masking, k_fold_index, make_reproducibility
from model_230505 import Attention, age_estimator

def age_pretrain(full_dataset, DEVICE, ind_list, SEED = None, N = 70, 
                 h_dim = 50, num_layers = 2, num_epoch = 200, batch_size = 16, 
                 loss_type = "MSE", learning_rate = 1e-3, weight_decay = 1e-5) : 

    if SEED is not None : 
        make_reproducibility(SEED)


    test_pred_list = torch.zeros(N)
    num_fold = len(ind_list)
    train_loss_list = [[] for _ in range(num_fold)]
    test_loss_list = [[] for _ in range(num_fold)]

    loss_function = nn.MSELoss().to(DEVICE)

    for fold in range(num_fold) : 
        print(f"{fold+1}th fold starting.")
        model = age_estimator(h_dim, num_layers).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

        I1, I2 = ind_list[fold]
        train_dataset = torch.utils.data.Subset(full_dataset, indices=I1)       
        test_dataset = torch.utils.data.Subset(full_dataset, indices=I2)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=I2.shape[0], shuffle=False)

        # test_prediction = None
        
        start = time.time()
        for epoch in tqdm(range(num_epoch)) : 
            model.train()
            for batch, (batch_data, batch_age, batch_mmse) in enumerate(train_loader) : 

                batch_data = batch_data.to(DEVICE)
                batch_age = batch_age.to(DEVICE)
                batch_mmse = batch_mmse.to(DEVICE)

                optimizer.zero_grad()
                train_loss = loss_function(model(batch_data, batch_mmse), batch_age)
                train_loss_list[fold].append(train_loss.item())
                
                train_loss.backward()
                optimizer.step()

            
            model.eval()
            for batch, (batch_data, batch_age, batch_mmse) in enumerate(test_loader) :

                batch_data = batch_data.to(DEVICE)
                batch_age = batch_age.to(DEVICE)
                batch_mmse = batch_mmse.to(DEVICE)
                
                test_loss = loss_function(model(batch_data, batch_mmse), batch_age)
                test_loss_list[fold].append(test_loss.item())

                if epoch == num_epoch - 1 : 
                    test_prediction = model(batch_data, batch_mmse).detach().cpu()

            # if (1+epoch) % (num_epoch // 5) == 0:  
            #     print(f"[Epoch {epoch:3d}] Train loss: {train_loss.item():.6f} Test loss: {test_loss.item():.6f}")
        
        end = time.time()
        # print("Time ellapsed in training is: {}".format(end - start))

        test_pred_list[I2] = test_prediction
    
    return test_pred_list, train_loss_list, test_loss_list



def patient_age_train(normal_dataset, patient_dataset, DEVICE, SEED = None, N = 80, patient_N = 80,  
                      h_dim = 50, num_layers = 2, num_epoch = 200, batch_size = 8, 
                      loss_type = "MSE", learning_rate = 1e-3, weight_decay = 1e-2) : 

    if SEED is not None : 
        make_reproducibility(SEED)

    train_loss_list = []
    test_loss_list = []

    loss_function = nn.MSELoss().to(DEVICE)
    module = age_estimator(h_dim, num_layers).to(DEVICE)
    optimizer = optim.Adam(module.parameters(), lr=learning_rate, weight_decay = weight_decay)

    train_loader = torch.utils.data.DataLoader(dataset=normal_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=patient_dataset, batch_size=patient_N, shuffle=False)

    start = time.time()
    for epoch in tqdm(range(num_epoch)) : 
        module.train()
        for batch, (batch_data, batch_age, batch_mmse) in enumerate(train_loader) : 

            batch_data = batch_data.to(DEVICE)
            batch_age = batch_age.to(DEVICE)
            batch_mmse = batch_mmse.to(DEVICE)

            optimizer.zero_grad()
            train_loss = loss_function(module(batch_data, batch_mmse), batch_age)
            train_loss_list.append(train_loss.item())

            train_loss.backward()
            optimizer.step()
        
        module.eval()
        for batch, (batch_data, batch_age, batch_mmse) in enumerate(test_loader) :

            batch_data = batch_data.to(DEVICE)
            batch_age = batch_age.to(DEVICE)
            batch_mmse = batch_mmse.to(DEVICE)
            
            test_loss = loss_function(module(batch_data, batch_mmse), batch_age)
            test_loss_list.append(test_loss.item())

            if epoch == num_epoch - 1 : 
                test_pred_list = module(batch_data, batch_mmse).detach().cpu()
    end = time.time()

    return test_pred_list, train_loss_list, test_loss_list
