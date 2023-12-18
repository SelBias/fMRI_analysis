
import os
import time
import torch
import random
import copy

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader

from util import MYTensorDataset, make_masking, make_arbitrary_masking, k_fold_index, make_reproducibility
from model import Attention, age_estimator
from loss import Binary_module, NLL_module, Proposed_module, loss_generator

def normal_age_pretrain(
    full_dataset, DEVICE, ind_list, SEED = None, N = 70, 
    h_dim = 50, num_layers = 2, num_epoch = 200, batch_size = 16, 
    loss_type = "MSE", learning_rate = 1e-3, weight_decay = 1e-5, patience = 10) : 

    if SEED is not None : 
        make_reproducibility(SEED)

    # num_fold = K in K-fold CV
    num_fold = len(ind_list)

    test_pred_list = torch.zeros(N)
    train_loss_list = [[] for _ in range(num_fold)]
    val_loss_list = [[] for _ in range(num_fold)]
    test_loss_list = [[] for _ in range(num_fold)]

    # Use MSE loss to train age prediction model
    loss_function = nn.MSELoss().to(DEVICE)

    # K-fold CV start
    for fold in range(num_fold) : 
        print(f"{fold+1}th fold starting.")

        # Initialize model and optimizer
        model = age_estimator(h_dim, num_layers).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

        # I1 : train indice (including validation dataset)
        # devided into train_indice and val_indice
        I1, test_indice = ind_list[fold]
        J1, J2 = k_fold_index(I1.shape[0], num_fold - 1)[0]
        train_indice = I1[J1]
        val_indice = I1[J2]

        # Split dataset into train/validation/test dataset
        train_dataset = torch.utils.data.Subset(full_dataset, indices=train_indice)    
        val_dataset = torch.utils.data.Subset(full_dataset, indices=val_indice)     
        test_dataset = torch.utils.data.Subset(full_dataset, indices=test_indice)

        # Define train/validation/test dataloader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=val_indice.shape[0], shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_indice.shape[0], shuffle=False)

        # Setup for early stopping
        best_model = copy.deepcopy(model)
        best_loss = 1e6
        update_count = 0
        update_stop = False
        
        for epoch in tqdm(range(num_epoch)) : 

            if update_stop is True : 
                break

            # training step
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

            # validation step
            model.eval()
            for batch, (batch_data, batch_age, batch_mmse) in enumerate(val_loader) : 

                batch_data = batch_data.to(DEVICE)
                batch_age = batch_age.to(DEVICE)
                batch_mmse = batch_mmse.to(DEVICE)

                validation_loss = loss_function(model(batch_data, batch_mmse), batch_age).item()
                val_loss_list[fold].append(validation_loss)
            
                if validation_loss < best_loss : 
                    best_loss = validation_loss
                    best_model = copy.deepcopy(model)
                    update_count = 0
                else : 
                    update_count += 1

                if update_count == patience : 
                    update_stop = True
                    print(f'{fold}th fold stopped training at {epoch}th epoch')

            # After training, predict test dataset
            best_model.eval()
            for batch, (batch_data, batch_age, batch_mmse) in enumerate(test_loader) :

                batch_data = batch_data.to(DEVICE)
                batch_age = batch_age.to(DEVICE)
                batch_mmse = batch_mmse.to(DEVICE)
                
                test_loss = loss_function(best_model(batch_data, batch_mmse), batch_age)
                test_loss_list[fold].append(test_loss.item())

                test_prediction = best_model(batch_data, batch_mmse).detach().cpu()

        test_pred_list[test_indice] = test_prediction
    
    return test_pred_list, train_loss_list, val_loss_list, test_loss_list



def patient_age_pretrain(normal_train_dataset, normal_val_dataset, patient_dataset, 
                      DEVICE, SEED = None, N = 70, val_N = 20, patient_N = 80,  
                      h_dim = 50, num_layers = 2, num_epoch = 200, batch_size = 8, 
                      loss_type = "MSE", learning_rate = 1e-3, weight_decay = 1e-2, patience = 10) : 

    if SEED is not None : 
        make_reproducibility(SEED)

    train_loss_list = []
    val_loss_list = []
    test_loss_list = []

    loss_function = nn.MSELoss().to(DEVICE)
    module = age_estimator(h_dim, num_layers).to(DEVICE)
    optimizer = optim.Adam(module.parameters(), lr=learning_rate, weight_decay = weight_decay)

    train_loader = torch.utils.data.DataLoader(dataset=normal_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=normal_val_dataset, batch_size=val_N, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=patient_dataset, batch_size=patient_N, shuffle=False)

    best_model = copy.deepcopy(module)
    best_loss = 1e6
    update_count = 0
    update_stop = False

    for epoch in tqdm(range(num_epoch)) : 
        if update_stop is True : 
            break

        # Training step
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
        
        # Validation step
        module.eval()
        for batch, (batch_data, batch_age, batch_mmse) in enumerate(val_loader) :
            batch_data = batch_data.to(DEVICE)
            batch_age = batch_age.to(DEVICE)
            batch_mmse = batch_mmse.to(DEVICE)

            validation_loss = loss_function(module(batch_data, batch_mmse), batch_age).item()
            val_loss_list.append(validation_loss)
            if validation_loss < best_loss : 
                best_loss = validation_loss
                best_model = copy.deepcopy(module)
                update_count = 0
            else : 
                update_count += 1

            if update_count == patience : 
                update_stop = True
                print(f'Patient age prediction stopped training at {epoch}th epoch')

        # Early stopping
        best_model.eval()
        for batch, (batch_data, batch_age, batch_mmse) in enumerate(test_loader) :

            batch_data = batch_data.to(DEVICE)
            batch_age = batch_age.to(DEVICE)
            batch_mmse = batch_mmse.to(DEVICE)
            
            test_loss = loss_function(best_model(batch_data, batch_mmse), batch_age)
            test_loss_list.append(test_loss.item())

            # if epoch == num_epoch - 1 | update_stop is True : 
            test_pred_list = best_model(batch_data, batch_mmse).detach().cpu()

    return test_pred_list, train_loss_list, val_loss_list, test_loss_list

# Patient detection with age-pretrain
def coupled_train(full_dataset, module_type, DEVICE, ind_list, 
                  SEED = None, N = 150, K = 4, 
                  h_dim = 50, num_layers = 2, num_epoch = 200, batch_size = 8, 
                  loss_type = 'Proposed', weight = None, p_star = None, eps = 1e-8, 
                  learning_rate = 1e-3, weight_decay = 1e-2, patience = 10) : 

    if SEED is not None : 
        make_reproducibility(SEED)

    test_pred_list_1 = torch.zeros([N, K])
    test_pred_list_2 = torch.zeros([N, K])
    if K == 2 : 
        test_pred_list_1 = torch.zeros(N)
        test_pred_list_2 = torch.zeros(N)
        loss_type = 'Binary'    

    num_fold = len(ind_list)



    train_loss_list_1 = [[] for _ in range(num_fold)]
    train_loss_list_2 = [[] for _ in range(num_fold)]

    val_loss_list_1 = [[] for _ in range(num_fold)]
    val_loss_list_2 = [[] for _ in range(num_fold)]

    test_loss_list_1 = [[] for _ in range(num_fold)]
    test_loss_list_2 = [[] for _ in range(num_fold)]
    
    loss_function = loss_generator(DEVICE, loss_type, K, weight, p_star, eps)


    for fold in range(num_fold) : 
        print(f"{fold+1}th fold starting.")
        module = module_type(h_dim, num_layers, K).to(DEVICE)
        optimizer = optim.Adam(module.parameters(), lr=learning_rate, weight_decay = weight_decay)

        I1, test_indice = ind_list[fold]
        J1, J2 = k_fold_index(I1.shape[0], num_fold - 1)[0]
        train_indice = I1[J1]
        val_indice = I1[J2]
        
        train_dataset = torch.utils.data.Subset(full_dataset, indices=train_indice)    
        val_dataset = torch.utils.data.Subset(full_dataset, indices=val_indice)     
        test_dataset = torch.utils.data.Subset(full_dataset, indices=test_indice)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=val_indice.shape[0], shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_indice.shape[0], shuffle=False)

        best_module_1 = copy.deepcopy(module)
        best_loss_1 = 1e6
        update_count_1 = 0
        update_stop_1 = False

        # Pre-training start
        for epoch in tqdm(range(num_epoch)) : 

            if update_stop_1 is True : 
                break

            # training step
            module.train()
            for batch, (batch_data, batch_class, batch_mmse, batch_age, batch_age_diff) in enumerate(train_loader) : 

                batch_data = batch_data.to(DEVICE)
                batch_class = batch_class.to(DEVICE)
                batch_mmse = batch_mmse.to(DEVICE)
                batch_age = batch_age.to(DEVICE)
                batch_age_diff = batch_age_diff.to(DEVICE)

                optimizer.zero_grad()
                train_loss = loss_function(module.pre_forward(batch_data, batch_mmse, batch_age, batch_age_diff), batch_class)
                train_loss_list_1[fold].append(train_loss.item())

                train_loss.backward()
                optimizer.step()
            
            module.eval()
            for batch, (batch_data, batch_class, batch_mmse, batch_age, batch_age_diff) in enumerate(val_loader) :

                batch_data = batch_data.to(DEVICE)
                batch_class = batch_class.to(DEVICE)
                batch_mmse = batch_mmse.to(DEVICE)
                batch_age = batch_age.to(DEVICE)
                batch_age_diff = batch_age_diff.to(DEVICE)
                
                validation_loss = loss_function(module.pre_forward(batch_data, batch_mmse, batch_age, batch_age_diff), batch_class).item()
                val_loss_list_1[fold].append(validation_loss)
                
                if validation_loss < best_loss_1 : 
                    best_loss_1 = validation_loss
                    best_module_1 = copy.deepcopy(module)
                    update_count_1 = 0
                else : 
                    update_count_1 += 1

                if update_count_1 == patience : 
                    update_stop_1 = True
                    print(f'{fold}th fold stopped training at {epoch}th epoch')

            best_module_1.eval()
            for batch, (batch_data, batch_class, batch_mmse, batch_age, batch_age_diff) in enumerate(test_loader) :

                batch_data = batch_data.to(DEVICE)
                batch_class = batch_class.to(DEVICE)
                batch_mmse = batch_mmse.to(DEVICE)
                batch_age = batch_age.to(DEVICE)
                batch_age_diff = batch_age_diff.to(DEVICE)
                
                test_loss = loss_function(best_module_1.pre_forward(batch_data, batch_mmse, batch_age, batch_age_diff), batch_class)
                test_loss_list_1[fold].append(test_loss.item())

                test_prediction = best_module_1.pre_forward(batch_data, batch_mmse, batch_age, batch_age_diff).detach().cpu()

        if K == 2 : 
            test_pred_list_1[test_indice] = test_prediction
        else :
            test_pred_list_1[test_indice] = test_prediction.squeeze(1)

        # training using age_difference
        best_module_2 = copy.deepcopy(module)
        best_loss_2 = 1e6
        update_count_2 = 0
        update_stop_2 = False
        for epoch in tqdm(range(num_epoch)) : 

            if update_stop_2 is True : 
                break

            module.train()
            for batch, (batch_data, batch_class, batch_mmse, batch_age, batch_age_diff) in enumerate(train_loader) : 

                batch_data = batch_data.to(DEVICE)
                batch_class = batch_class.to(DEVICE)
                batch_mmse = batch_mmse.to(DEVICE)
                batch_age = batch_age.to(DEVICE)
                batch_age_diff = batch_age_diff.to(DEVICE)

                optimizer.zero_grad()
                train_loss = loss_function(module(batch_data, batch_mmse, batch_age, batch_age_diff), batch_class)
                train_loss_list_2[fold].append(train_loss.item())

                train_loss.backward()
                optimizer.step()
            
            module.eval()
            for batch, (batch_data, batch_class, batch_mmse, batch_age, batch_age_diff) in enumerate(val_loader) :

                batch_data = batch_data.to(DEVICE)
                batch_class = batch_class.to(DEVICE)
                batch_mmse = batch_mmse.to(DEVICE)
                batch_age = batch_age.to(DEVICE)
                batch_age_diff = batch_age_diff.to(DEVICE)
                
                validation_loss = loss_function(module(batch_data, batch_mmse, batch_age, batch_age_diff), batch_class).item()
                val_loss_list_1[fold].append(validation_loss)
                
                if validation_loss < best_loss_2 : 
                    best_loss_2 = validation_loss
                    best_module_2 = copy.deepcopy(module)
                    update_count_2 = 0
                else : 
                    update_count_2 += 1

                if update_count_2 == patience : 
                    update_stop_2 = True
                    print(f'{fold}th fold stopped training at {epoch}th epoch')
            
            best_module_2.eval()
            for batch, (batch_data, batch_class, batch_mmse, batch_age, batch_age_diff) in enumerate(test_loader) :

                batch_data = batch_data.to(DEVICE)
                batch_class = batch_class.to(DEVICE)
                batch_mmse = batch_mmse.to(DEVICE)
                batch_age = batch_age.to(DEVICE)
                batch_age_diff = batch_age_diff.to(DEVICE)
                
                test_loss = loss_function(best_module_2(batch_data, batch_mmse, batch_age, batch_age_diff), batch_class).item()
                test_loss_list_2[fold].append(test_loss)

                test_prediction = best_module_2(batch_data, batch_mmse, batch_age, batch_age_diff).detach().cpu()

        if K == 2 : 
            test_pred_list_2[test_indice] = test_prediction
        else :
            test_pred_list_2[test_indice] = test_prediction.squeeze(1)
    
    return [test_pred_list_1, train_loss_list_1, val_loss_list_1, test_loss_list_1], [test_pred_list_2, train_loss_list_2, val_loss_list_2, test_loss_list_2]


