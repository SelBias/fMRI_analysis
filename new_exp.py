
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

from new_util import TensorDataset, make_reproducibility, make_decision, multiclass_accuracy
from new_model import Attention, age_estimator, classifier
from new_loss import proposed_loss

def nll_pretrain(
    full_dataset, train_indice, val_indice, test_indice, DEVICE, SEED, 
    N = 150, K = 3, num_fold = 10, 
    h_dim = 50, num_layers = 2, num_epoch = 200, batch_size = 8,
    weight = None, p_star = None, eps = 1e-8,
    learning_rate = 1e-3, weight_decay = 1e-2, patience = 10, experiment_name='exp') : 

    test_prediction = torch.zeros([N, K])
    test_decision = torch.zeros(N, dtype=torch.int32)

    acc_list = []

    train_nll_list = [[] for _ in range(num_fold)]
    train_pnll_list = [[] for _ in range(num_fold)]

    val_nll_list = [[] for _ in range(num_fold)]
    val_pnll_list = [[] for _ in range(num_fold)]

    test_nll_list = [[] for _ in range(num_fold)]
    test_pnll_list = [[] for _ in range(num_fold)]

    nll_loss = proposed_loss(DEVICE, K, weight)
    pnll_loss = proposed_loss(DEVICE, K, weight, p_star, eps)

    for fold in range(num_fold) : 
        print(f"{fold}th fold starting.")
        make_reproducibility(SEED + fold)

        model = classifier(h_dim, num_layers, K).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

        train_dataset = torch.utils.data.Subset(full_dataset, indices=train_indice[fold])
        val_dataset = torch.utils.data.Subset(full_dataset, indices=val_indice[fold])
        test_dataset = torch.utils.data.Subset(full_dataset, indices=test_indice[fold])

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=15, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=15, shuffle=False)

        best_loss = 1e8
        best_model = copy.deepcopy(model)
        best_optim = copy.deepcopy(optimizer)
        update_count = 0
        update_stop = False

        for epoch in tqdm(range(num_epoch)) : 
            if update_stop : 
                break

            model.train()
            for _, (batch_data, batch_class, batch_mmse, batch_age, _) in enumerate(train_loader) :

                batch_data = batch_data.to(DEVICE)
                batch_class = batch_class.to(DEVICE)
                batch_mmse = batch_mmse.to(DEVICE)
                batch_age = batch_age.to(DEVICE)

                optimizer.zero_grad()
                train_nll_loss = nll_loss(model.pre_forward(batch_data, batch_mmse, batch_age), batch_class)
                train_pnll_loss = pnll_loss(model.pre_forward(batch_data, batch_mmse, batch_age), batch_class)

                train_nll_list[fold].append(train_nll_loss.item())
                train_pnll_list[fold].append(train_pnll_loss.item())

                train_nll_loss.backward()
                optimizer.step()


            model.eval()
            for _, (batch_data, batch_class, batch_mmse, batch_age, _) in enumerate(val_loader) :

                batch_data = batch_data.to(DEVICE)
                batch_class = batch_class.to(DEVICE)
                batch_mmse = batch_mmse.to(DEVICE)
                batch_age = batch_age.to(DEVICE)

                val_nll_loss = nll_loss(model.pre_forward(batch_data, batch_mmse, batch_age), batch_class).item()
                val_pnll_loss = pnll_loss(model.pre_forward(batch_data, batch_mmse, batch_age), batch_class).item()

                val_nll_list[fold].append(val_nll_loss)
                val_pnll_list[fold].append(val_pnll_loss)

                if val_nll_loss < best_loss :
                    best_loss = val_nll_loss
                    best_model = copy.deepcopy(model)
                    best_optim = copy.deepcopy(optimizer)
                    update_count = 0
                else :
                    update_count += 1

                if update_count == patience :
                    update_stop = True
                    print(f'{fold}th fold stopped training at {epoch - patience}th epoch')


            model.eval()
            for _, (batch_data, batch_class, batch_mmse, batch_age, _) in enumerate(test_loader) :

                batch_data = batch_data.to(DEVICE)
                batch_class = batch_class.to(DEVICE)
                batch_mmse = batch_mmse.to(DEVICE)
                batch_age = batch_age.to(DEVICE)

                test_nll_loss = nll_loss(model.pre_forward(batch_data, batch_mmse, batch_age), batch_class)
                test_pnll_loss = pnll_loss(model.pre_forward(batch_data, batch_mmse, batch_age), batch_class)
                
                test_nll_list[fold].append(test_nll_loss.item())
                test_pnll_list[fold].append(test_pnll_loss.item())


        best_model.eval()
        for _, (batch_data, batch_class, batch_mmse, batch_age, _) in enumerate(test_loader) :

            batch_data = batch_data.to(DEVICE)
            # batch_class = batch_class.to(DEVICE)
            batch_mmse = batch_mmse.to(DEVICE)
            batch_age = batch_age.to(DEVICE)

            fold_pred = best_model.pre_forward(batch_data, batch_mmse, batch_age).detach()

            test_nll_loss = nll_loss(fold_pred, batch_class).item()
            test_pnll_loss = pnll_loss(fold_pred, batch_class).item()

            fold_pred = fold_pred.cpu()

            fold_decision = make_decision(fold_pred, K = K)
            fold_acc = multiclass_accuracy(fold_decision, batch_class)
            print(f'{fold}-th fold test NLL, PNLL, accuracy : {test_nll_loss:.4f}, {test_pnll_loss:.4f}, {fold_acc:.4f}')

        acc_list.append(fold_acc)

        test_decision[test_indice[fold]] = fold_decision.int()
        test_prediction[test_indice[fold]] = fold_pred

        # Checkpoint 1
        checkpoint = {
            'model': best_model.state_dict(),
            'optimizer': best_optim.state_dict() 
        }

        os.makedirs(f'./{experiment_name}', exist_ok=True)
        torch.save(checkpoint, f'./{experiment_name}/NLL_{fold}_pretrain.pth')


    nll_loss = proposed_loss(torch.device('cpu'), K, weight)
    pnll_loss = proposed_loss(torch.device('cpu'), K, weight, p_star, eps)
    total_nll = nll_loss(test_prediction, full_dataset.tensors[1]).item()
    total_pnll = pnll_loss(test_prediction, full_dataset.tensors[1]).item()
    total_acc = multiclass_accuracy(test_decision, full_dataset.tensors[1])

    print(f'Total NLL, PNLL, accuracy : {total_nll:.4f}, {total_pnll:.4f}, {total_acc:.4f}')

    np.save(f'./{experiment_name}/NLL_prediction', test_prediction)
    np.save(f'./{experiment_name}/NLL_decision', test_decision)
    np.save(f'./{experiment_name}/NLL_accuracy', acc_list)
    
    np.save(f'./{experiment_name}/NLL_train_nll', train_nll_list)
    np.save(f'./{experiment_name}/NLL_train_pnll', train_pnll_list)
    np.save(f'./{experiment_name}/NLL_val_nll', val_nll_list)
    np.save(f'./{experiment_name}/NLL_val_pnll', val_pnll_list)
    np.save(f'./{experiment_name}/NLL_test_nll', test_nll_list)
    np.save(f'./{experiment_name}/NLL_test_pnll', test_pnll_list)

    return [test_prediction, 
            train_nll_list, train_pnll_list, 
            val_nll_list, val_pnll_list, 
            test_nll_list, test_pnll_list]



def pnll_train(
    full_dataset, train_indice, val_indice, test_indice, DEVICE, SEED, 
    load_dir = './',
    N = 150, K = 3, num_fold = 10, 
    h_dim = 50, num_layers = 2, num_epoch = 200, batch_size = 8,
    weight = None, p_star = None, eps = 1e-8,
    learning_rate = 1e-3, weight_decay = 1e-2, patience = 10, experiment_name='exp') : 
    
    test_prediction = torch.zeros([N, K])
    test_decision = torch.zeros(N, dtype=torch.int32)

    acc_list = []

    train_nll_list = [[] for _ in range(num_fold)]
    train_pnll_list = [[] for _ in range(num_fold)]

    val_nll_list = [[] for _ in range(num_fold)]
    val_pnll_list = [[] for _ in range(num_fold)]

    test_nll_list = [[] for _ in range(num_fold)]
    test_pnll_list = [[] for _ in range(num_fold)]

    nll_loss = proposed_loss(DEVICE, K, weight)
    pnll_loss = proposed_loss(DEVICE, K, weight, p_star, eps)
    
    for fold in range(num_fold) : 
        print(f"{fold}th fold starting.")
        make_reproducibility(SEED + fold)

        model = classifier(h_dim, num_layers, K).to(DEVICE)
        optimizer = optim.Adam(model.last_layer1.parameters(), lr=learning_rate, weight_decay = weight_decay)

        # train_dataset = torch.utils.data.Subset(full_dataset, indices=train_indice[fold])
        # val_dataset = torch.utils.data.Subset(full_dataset, indices=val_indice[fold])
        # test_dataset = torch.utils.data.Subset(full_dataset, indices=test_indice[fold])
        # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        # val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=15, shuffle=False)
        # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=15, shuffle=False)

        # torch_dict = torch.load(f'{load_dir}/NLL_{fold}_pretrain.pth')
        # model.load_state_dict(torch_dict['model'])
        model.load_state_dict(torch.load(f'{load_dir}/NLL_{fold}_pretrain.pth')['model'])

        model.eval()
        with torch.no_grad() :  
            train_X = model.get_last_hidden_layer(full_dataset.tensors[0][train_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[2][train_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[3][train_indice[fold]].to(DEVICE)).detach()
            val_X = model.get_last_hidden_layer(full_dataset.tensors[0][val_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[2][val_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[3][val_indice[fold]].to(DEVICE)).detach()
            test_X =  model.get_last_hidden_layer(full_dataset.tensors[0][test_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[2][test_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[3][test_indice[fold]].to(DEVICE)).detach()
        train_class = full_dataset.tensors[1][train_indice[fold]].to(DEVICE)
        val_class = full_dataset.tensors[1][val_indice[fold]].to(DEVICE)
        test_class = full_dataset.tensors[1][test_indice[fold]].to(DEVICE)
        
        best_loss = 1e8
        best_model = copy.deepcopy(model)
        best_optim = copy.deepcopy(optimizer)
        update_count = 0
        update_stop = False

        for epoch in tqdm(range(num_epoch)) : 
            if update_stop : 
                break

            model.train()
            for _ in range(15) : 

                optimizer.zero_grad()
                train_nll_loss = nll_loss(F.softmax(model.last_layer1(train_X)), train_class)
                train_pnll_loss = pnll_loss(F.softmax(model.last_layer1(train_X)), train_class)

                train_nll_list[fold].append(train_nll_loss.item())
                train_pnll_list[fold].append(train_pnll_loss.item())

                train_pnll_loss.backward()
                optimizer.step()


            model.eval()
            val_nll_loss = nll_loss(F.softmax(model.last_layer1(val_X)), val_class).item()
            val_pnll_loss = pnll_loss(F.softmax(model.last_layer1(val_X)), val_class).item()

            val_nll_list[fold].append(val_nll_loss)
            val_pnll_list[fold].append(val_pnll_loss)

            if val_pnll_loss < best_loss :
                best_loss = val_pnll_loss
                best_model = copy.deepcopy(model)
                best_optim = copy.deepcopy(optimizer)
                update_count = 0
            else :
                update_count += 1

            if update_count == patience :
                update_stop = True
                print(f'{fold}th fold stopped training at {epoch - patience}th epoch')


            model.eval()


            test_nll_loss  = nll_loss(F.softmax(model.last_layer1(test_X)), test_class)
            test_pnll_loss = pnll_loss(F.softmax(model.last_layer1(test_X)), test_class)
            
            test_nll_list[fold].append(test_nll_loss.item())
            test_pnll_list[fold].append(test_pnll_loss.item())


        best_model.eval()
        fold_pred = F.softmax(best_model.last_layer1(test_X)).detach()

        test_nll_loss = nll_loss(fold_pred, test_class).item()
        test_pnll_loss = pnll_loss(fold_pred, test_class).item()

        # fold_pred = fold_pred.cpu()

        fold_decision = make_decision(fold_pred, K = K).int()
        fold_acc = multiclass_accuracy(fold_decision, test_class)
        print(f'{fold}-th fold test NLL, PNLL, accuracy : {test_nll_loss:.4f}, {test_pnll_loss:.4f}, {fold_acc:.4f}')

        acc_list.append(fold_acc)

        test_decision[test_indice[fold]] = fold_decision.cpu()
        test_prediction[test_indice[fold]] = fold_pred.cpu()

        # Checkpoint 1
        checkpoint = {
            'model': best_model.state_dict(),
            'optimizer': best_optim.state_dict() 
        }

        os.makedirs(f'./{experiment_name}', exist_ok=True)
        torch.save(checkpoint, f'./{experiment_name}/PNLL_{fold}_pretrain.pth')


    nll_loss = proposed_loss(torch.device('cpu'), K, weight)
    pnll_loss = proposed_loss(torch.device('cpu'), K, weight, p_star, eps)
    total_nll = nll_loss(torch.as_tensor(test_prediction), full_dataset.tensors[1]).item()
    total_pnll = pnll_loss(torch.as_tensor(test_prediction), full_dataset.tensors[1]).item()
    total_acc = multiclass_accuracy(test_decision, full_dataset.tensors[1])

    print(f'Total NLL, PNLL, accuracy : {total_nll:.4f}, {total_pnll:.4f}, {total_acc:.4f}')

    np.save(f'./{experiment_name}/PNLL_prediction', test_prediction)
    np.save(f'./{experiment_name}/PNLL_decision', test_decision)
    np.save(f'./{experiment_name}/PNLL_accuracy', acc_list)
    
    np.save(f'./{experiment_name}/PNLL_train_nll', train_nll_list)
    np.save(f'./{experiment_name}/PNLL_train_pnll', train_pnll_list)
    np.save(f'./{experiment_name}/PNLL_val_nll', val_nll_list)
    np.save(f'./{experiment_name}/PNLL_val_pnll', val_pnll_list)
    np.save(f'./{experiment_name}/PNLL_test_nll', test_nll_list)
    np.save(f'./{experiment_name}/PNLL_test_pnll', test_pnll_list)

    return [test_prediction, 
            train_nll_list, train_pnll_list, 
            val_nll_list, val_pnll_list, 
            test_nll_list, test_pnll_list]





def nll_age_train(
    full_dataset, train_indice, val_indice, test_indice, DEVICE, SEED, 
    load_dir = './',
    N = 150, K = 3, num_fold = 10, 
    h_dim = 50, num_layers = 2, num_epoch = 200, batch_size = 8,
    weight = None, p_star = None, eps = 1e-8,
    learning_rate = 1e-3, weight_decay = 1e-2, patience = 10, experiment_name='exp') : 
    
    test_prediction = torch.zeros([N, K])
    test_decision = torch.zeros(N, dtype=torch.int32)

    acc_list = []

    train_nll_list = [[] for _ in range(num_fold)]
    train_pnll_list = [[] for _ in range(num_fold)]

    val_nll_list = [[] for _ in range(num_fold)]
    val_pnll_list = [[] for _ in range(num_fold)]

    test_nll_list = [[] for _ in range(num_fold)]
    test_pnll_list = [[] for _ in range(num_fold)]

    nll_loss = proposed_loss(DEVICE, K, weight)
    pnll_loss = proposed_loss(DEVICE, K, weight, p_star, eps)
    
    for fold in range(num_fold) : 
        print(f"{fold}th fold starting.")
        make_reproducibility(SEED + fold)

        model = classifier(h_dim, num_layers, K).to(DEVICE)
        optimizer = optim.Adam(model.last_layer1.parameters(), lr=learning_rate, weight_decay = weight_decay)

        train_dataset = torch.utils.data.Subset(full_dataset, indices=train_indice[fold])
        val_dataset = torch.utils.data.Subset(full_dataset, indices=val_indice[fold])
        test_dataset = torch.utils.data.Subset(full_dataset, indices=test_indice[fold])

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=15, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=15, shuffle=False)

        # torch_dict = torch.load(f'{load_dir}/NLL_{fold}_pretrain.pth')
        # model.load_state_dict(torch_dict['model'])
        model.load_state_dict(torch.load(f'{load_dir}/NLL_{fold}_pretrain.pth')['model'])
        
        best_loss = 1e8
        best_model = copy.deepcopy(model)
        best_optim = copy.deepcopy(optimizer)
        update_count = 0
        update_stop = False

        for epoch in tqdm(range(num_epoch)) : 
            if update_stop : 
                break

            model.train()
            for _, (batch_data, batch_class, batch_mmse, batch_age, batch_age_diff) in enumerate(train_loader) :

                batch_data = batch_data.to(DEVICE)
                batch_class = batch_class.to(DEVICE)
                batch_mmse = batch_mmse.to(DEVICE)
                batch_age = batch_age.to(DEVICE)
                batch_age_diff = batch_age_diff.to(DEVICE)

                optimizer.zero_grad()
                train_nll_loss = nll_loss(model(batch_data, batch_mmse, batch_age, batch_age_diff), batch_class)
                train_pnll_loss = pnll_loss(model(batch_data, batch_mmse, batch_age, batch_age_diff), batch_class)

                train_nll_list[fold].append(train_nll_loss.item())
                train_pnll_list[fold].append(train_pnll_loss.item())

                train_nll_loss.backward()
                optimizer.step()


            model.eval()
            for _, (batch_data, batch_class, batch_mmse, batch_age, batch_age_diff) in enumerate(val_loader) :

                batch_data = batch_data.to(DEVICE)
                batch_class = batch_class.to(DEVICE)
                batch_mmse = batch_mmse.to(DEVICE)
                batch_age = batch_age.to(DEVICE)
                batch_age_diff = batch_age_diff.to(DEVICE)

                val_nll_loss = nll_loss(model(batch_data, batch_mmse, batch_age, batch_age_diff), batch_class).item()
                val_pnll_loss = pnll_loss(model(batch_data, batch_mmse, batch_age, batch_age_diff), batch_class).item()

                val_nll_list[fold].append(val_nll_loss)
                val_pnll_list[fold].append(val_pnll_loss)

                if val_nll_loss < best_loss :
                    best_loss = val_nll_loss
                    best_model = copy.deepcopy(model)
                    best_optim = copy.deepcopy(optimizer)
                    update_count = 0
                else :
                    update_count += 1

                if update_count == patience :
                    update_stop = True
                    print(f'{fold}th fold stopped training at {epoch - patience}th epoch')


            model.eval()
            for _, (batch_data, batch_class, batch_mmse, batch_age, batch_age_diff) in enumerate(test_loader) :

                batch_data = batch_data.to(DEVICE)
                batch_class = batch_class.to(DEVICE)
                batch_mmse = batch_mmse.to(DEVICE)
                batch_age = batch_age.to(DEVICE)
                batch_age_diff = batch_age_diff.to(DEVICE)

                test_nll_loss = nll_loss(model(batch_data, batch_mmse, batch_age, batch_age_diff), batch_class)
                test_pnll_loss = pnll_loss(model(batch_data, batch_mmse, batch_age, batch_age_diff), batch_class)
                
                test_nll_list[fold].append(test_nll_loss.item())
                test_pnll_list[fold].append(test_pnll_loss.item())


        best_model.eval()
        for _, (batch_data, batch_class, batch_mmse, batch_age, batch_age_diff) in enumerate(test_loader) :

            batch_data = batch_data.to(DEVICE)
            # batch_class = batch_class.to(DEVICE)
            batch_mmse = batch_mmse.to(DEVICE)
            batch_age = batch_age.to(DEVICE)
            batch_age_diff = batch_age_diff.to(DEVICE)

            fold_pred = best_model(batch_data, batch_mmse, batch_age, batch_age_diff).detach()

            test_nll_loss = nll_loss(fold_pred, batch_class).item()
            test_pnll_loss = pnll_loss(fold_pred, batch_class).item()

            fold_pred = fold_pred.cpu()

            fold_decision = make_decision(fold_pred, K = K).int()
            fold_acc = multiclass_accuracy(fold_decision, batch_class)
            print(f'{fold}-th fold test NLL, PNLL, accuracy : {test_nll_loss:.4f}, {test_pnll_loss:.4f}, {fold_acc:.4f}')

        acc_list.append(fold_acc)

        test_decision[test_indice[fold]] = fold_decision.cpu()
        test_prediction[test_indice[fold]] = fold_pred.cpu()

        # Checkpoint 1
        checkpoint = {
            'model': best_model.state_dict(),
            'optimizer': best_optim.state_dict() 
        }

        os.makedirs(f'./{experiment_name}', exist_ok=True)
        torch.save(checkpoint, f'./{experiment_name}/NLL_age_{fold}_pretrain.pth')


    nll_loss = proposed_loss(torch.device('cpu'), K, weight)
    pnll_loss = proposed_loss(torch.device('cpu'), K, weight, p_star, eps)
    total_nll = nll_loss(torch.as_tensor(test_prediction), full_dataset.tensors[1]).item()
    total_pnll = pnll_loss(torch.as_tensor(test_prediction), full_dataset.tensors[1]).item()
    total_acc = multiclass_accuracy(test_decision, full_dataset.tensors[1])

    print(f'Total NLL, PNLL, accuracy : {total_nll:.4f}, {total_pnll:.4f}, {total_acc:.4f}')

    np.save(f'./{experiment_name}/NLL_age_prediction', test_prediction)
    np.save(f'./{experiment_name}/NLL_age_decision', test_decision)
    np.save(f'./{experiment_name}/NLL_age_accuracy', acc_list)
    
    np.save(f'./{experiment_name}/NLL_age_train_nll', train_nll_list)
    np.save(f'./{experiment_name}/NLL_age_train_pnll', train_pnll_list)
    np.save(f'./{experiment_name}/NLL_age_val_nll', val_nll_list)
    np.save(f'./{experiment_name}/NLL_age_val_pnll', val_pnll_list)
    np.save(f'./{experiment_name}/NLL_age_test_nll', test_nll_list)
    np.save(f'./{experiment_name}/NLL_age_test_pnll', test_pnll_list)

    return [test_prediction, 
            train_nll_list, train_pnll_list, 
            val_nll_list, val_pnll_list, 
            test_nll_list, test_pnll_list]






def pnll_age_train(
    full_dataset, train_indice, val_indice, test_indice, DEVICE, SEED, 
    load_dir = './',
    N = 150, K = 3, num_fold = 10, 
    h_dim = 50, num_layers = 2, num_epoch = 200, batch_size = 8,
    weight = None, p_star = None, eps = 1e-8,
    learning_rate = 1e-3, weight_decay = 1e-2, patience = 10, experiment_name='exp') : 
    
    test_prediction = torch.zeros([N, K])
    test_decision = torch.zeros(N, dtype=torch.int32)

    acc_list = []

    train_nll_list = [[] for _ in range(num_fold)]
    train_pnll_list = [[] for _ in range(num_fold)]

    val_nll_list = [[] for _ in range(num_fold)]
    val_pnll_list = [[] for _ in range(num_fold)]

    test_nll_list = [[] for _ in range(num_fold)]
    test_pnll_list = [[] for _ in range(num_fold)]

    nll_loss = proposed_loss(DEVICE, K, weight)
    pnll_loss = proposed_loss(DEVICE, K, weight, p_star, eps)
    
    for fold in range(num_fold) : 
        print(f"{fold}th fold starting.")
        make_reproducibility(SEED + fold)

        model = classifier(h_dim, num_layers, K).to(DEVICE)
        optimizer = optim.Adam(list(model.last_layer1.parameters()) + 
                               list(model.fc2.parameters()) + 
                               list(model.last_layer2.parameters()), lr=learning_rate, weight_decay = weight_decay)

        # train_dataset = torch.utils.data.Subset(full_dataset, indices=train_indice[fold])
        # val_dataset = torch.utils.data.Subset(full_dataset, indices=val_indice[fold])
        # test_dataset = torch.utils.data.Subset(full_dataset, indices=test_indice[fold])
        # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        # val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=15, shuffle=False)
        # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=15, shuffle=False)

        # torch_dict = torch.load(f'{load_dir}/NLL_{fold}_pretrain.pth')
        # model.load_state_dict(torch_dict['model'])
        model.load_state_dict(torch.load(f'{load_dir}/NLL_{fold}_pretrain.pth')['model'])

        model.eval()
        with torch.no_grad() :  
            train_X = model.get_last_hidden_layer(full_dataset.tensors[0][train_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[2][train_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[3][train_indice[fold]].to(DEVICE)).detach()
            val_X = model.get_last_hidden_layer(full_dataset.tensors[0][val_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[2][val_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[3][val_indice[fold]].to(DEVICE)).detach()
            test_X =  model.get_last_hidden_layer(full_dataset.tensors[0][test_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[2][test_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[3][test_indice[fold]].to(DEVICE)).detach()
        train_class = full_dataset.tensors[1][train_indice[fold]].to(DEVICE)
        val_class = full_dataset.tensors[1][val_indice[fold]].to(DEVICE)
        test_class = full_dataset.tensors[1][test_indice[fold]].to(DEVICE)
        
        train_age_diff = full_dataset.tensors[4][train_indice[fold]].to(DEVICE).unsqueeze(1)
        val_age_diff = full_dataset.tensors[4][val_indice[fold]].to(DEVICE).unsqueeze(1)
        test_age_diff = full_dataset.tensors[4][test_indice[fold]].to(DEVICE).unsqueeze(1)
        
        best_loss = 1e8
        best_model = copy.deepcopy(model)
        best_optim = copy.deepcopy(optimizer)
        update_count = 0
        update_stop = False

        for epoch in tqdm(range(num_epoch)) : 
            if update_stop : 
                break

            model.train()
            for _ in range(15) : 

                optimizer.zero_grad()
                train_nll_loss = nll_loss(F.softmax(model.last_layer1(train_X) + model.last_layer2(model.fc2(train_age_diff))), train_class)
                train_pnll_loss = pnll_loss(F.softmax(model.last_layer1(train_X) + model.last_layer2(model.fc2(train_age_diff))), train_class)

                train_nll_list[fold].append(train_nll_loss.item())
                train_pnll_list[fold].append(train_pnll_loss.item())

                train_pnll_loss.backward()
                optimizer.step()


            model.eval()
            val_nll_loss = nll_loss(F.softmax(model.last_layer1(val_X) + model.last_layer2(model.fc2(val_age_diff))), val_class).item()
            val_pnll_loss = pnll_loss(F.softmax(model.last_layer1(val_X) + model.last_layer2(model.fc2(val_age_diff))), val_class).item()

            val_nll_list[fold].append(val_nll_loss)
            val_pnll_list[fold].append(val_pnll_loss)

            if val_pnll_loss < best_loss :
                best_loss = val_pnll_loss
                best_model = copy.deepcopy(model)
                best_optim = copy.deepcopy(optimizer)
                update_count = 0
            else :
                update_count += 1

            if update_count == patience :
                update_stop = True
                print(f'{fold}th fold stopped training at {epoch - patience}th epoch')


            model.eval()


            test_nll_loss  = nll_loss(F.softmax(model.last_layer1(test_X) + model.last_layer2(model.fc2(test_age_diff))), test_class)
            test_pnll_loss = pnll_loss(F.softmax(model.last_layer1(test_X) + model.last_layer2(model.fc2(test_age_diff))), test_class)
            
            test_nll_list[fold].append(test_nll_loss.item())
            test_pnll_list[fold].append(test_pnll_loss.item())


        best_model.eval()
        fold_pred = F.softmax(best_model.last_layer1(test_X) + best_model.last_layer2(best_model.fc2(test_age_diff))).detach()

        test_nll_loss = nll_loss(fold_pred, test_class).item()
        test_pnll_loss = pnll_loss(fold_pred, test_class).item()

        # fold_pred = fold_pred.cpu()

        fold_decision = make_decision(fold_pred, K = K).int()
        fold_acc = multiclass_accuracy(fold_decision, test_class)
        print(f'{fold}-th fold test NLL, PNLL, accuracy : {test_nll_loss:.4f}, {test_pnll_loss:.4f}, {fold_acc:.4f}')

        acc_list.append(fold_acc)

        test_decision[test_indice[fold]] = fold_decision.cpu()
        test_prediction[test_indice[fold]] = fold_pred.cpu()

        # Checkpoint 1
        checkpoint = {
            'model': best_model.state_dict(),
            'optimizer': best_optim.state_dict() 
        }

        os.makedirs(f'./{experiment_name}', exist_ok=True)
        torch.save(checkpoint, f'./{experiment_name}/PNLL_age_{fold}_pretrain.pth')


    nll_loss = proposed_loss(torch.device('cpu'), K, weight)
    pnll_loss = proposed_loss(torch.device('cpu'), K, weight, p_star, eps)
    total_nll = nll_loss(torch.as_tensor(test_prediction), full_dataset.tensors[1]).item()
    total_pnll = pnll_loss(torch.as_tensor(test_prediction), full_dataset.tensors[1]).item()
    total_acc = multiclass_accuracy(test_decision, full_dataset.tensors[1])

    print(f'Total NLL, PNLL, accuracy : {total_nll:.4f}, {total_pnll:.4f}, {total_acc:.4f}')

    np.save(f'./{experiment_name}/PNLL_age_prediction', test_prediction)
    np.save(f'./{experiment_name}/PNLL_age_decision', test_decision)
    np.save(f'./{experiment_name}/PNLL_age_accuracy', acc_list)
    
    np.save(f'./{experiment_name}/PNLL_age_train_nll', train_nll_list)
    np.save(f'./{experiment_name}/PNLL_age_train_pnll', train_pnll_list)
    np.save(f'./{experiment_name}/PNLL_age_val_nll', val_nll_list)
    np.save(f'./{experiment_name}/PNLL_age_val_pnll', val_pnll_list)
    np.save(f'./{experiment_name}/PNLL_age_test_nll', test_nll_list)
    np.save(f'./{experiment_name}/PNLL_age_test_pnll', test_pnll_list)

    return [test_prediction, 
            train_nll_list, train_pnll_list, 
            val_nll_list, val_pnll_list, 
            test_nll_list, test_pnll_list]




def pnll_age_train_from_pnll(
    full_dataset, train_indice, val_indice, test_indice, DEVICE, SEED, 
    load_dir = './',
    N = 150, K = 3, num_fold = 10, 
    h_dim = 50, num_layers = 2, num_epoch = 200, batch_size = 8,
    weight = None, p_star = None, eps = 1e-8,
    learning_rate = 1e-3, weight_decay = 1e-2, patience = 10, experiment_name='exp') : 
    
    test_prediction = torch.zeros([N, K])
    test_decision = torch.zeros(N, dtype=torch.int32)

    acc_list = []

    train_nll_list = [[] for _ in range(num_fold)]
    train_pnll_list = [[] for _ in range(num_fold)]

    val_nll_list = [[] for _ in range(num_fold)]
    val_pnll_list = [[] for _ in range(num_fold)]

    test_nll_list = [[] for _ in range(num_fold)]
    test_pnll_list = [[] for _ in range(num_fold)]

    nll_loss = proposed_loss(DEVICE, K, weight)
    pnll_loss = proposed_loss(DEVICE, K, weight, p_star, eps)
    
    for fold in range(num_fold) : 
        print(f"{fold}th fold starting.")
        make_reproducibility(SEED + fold)

        model = classifier(h_dim, num_layers, K).to(DEVICE)
        optimizer = optim.Adam(list(model.last_layer1.parameters()) + 
                               list(model.fc2.parameters()) + 
                               list(model.last_layer2.parameters()), lr=learning_rate, weight_decay = weight_decay)

        # train_dataset = torch.utils.data.Subset(full_dataset, indices=train_indice[fold])
        # val_dataset = torch.utils.data.Subset(full_dataset, indices=val_indice[fold])
        # test_dataset = torch.utils.data.Subset(full_dataset, indices=test_indice[fold])
        # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        # val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=15, shuffle=False)
        # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=15, shuffle=False)

        # torch_dict = torch.load(f'{load_dir}/NLL_{fold}_pretrain.pth')
        # model.load_state_dict(torch_dict['model'])
        model.load_state_dict(torch.load(f'{load_dir}/PNLL_{fold}_pretrain.pth')['model'])

        model.eval()
        with torch.no_grad() :  
            train_X = model.get_last_hidden_layer(full_dataset.tensors[0][train_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[2][train_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[3][train_indice[fold]].to(DEVICE)).detach()
            val_X = model.get_last_hidden_layer(full_dataset.tensors[0][val_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[2][val_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[3][val_indice[fold]].to(DEVICE)).detach()
            test_X =  model.get_last_hidden_layer(full_dataset.tensors[0][test_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[2][test_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[3][test_indice[fold]].to(DEVICE)).detach()
        train_class = full_dataset.tensors[1][train_indice[fold]].to(DEVICE)
        val_class = full_dataset.tensors[1][val_indice[fold]].to(DEVICE)
        test_class = full_dataset.tensors[1][test_indice[fold]].to(DEVICE)
        
        train_age_diff = full_dataset.tensors[4][train_indice[fold]].to(DEVICE).unsqueeze(1)
        val_age_diff = full_dataset.tensors[4][val_indice[fold]].to(DEVICE).unsqueeze(1)
        test_age_diff = full_dataset.tensors[4][test_indice[fold]].to(DEVICE).unsqueeze(1)
        
        best_loss = 1e8
        best_model = copy.deepcopy(model)
        best_optim = copy.deepcopy(optimizer)
        update_count = 0
        update_stop = False

        for epoch in tqdm(range(num_epoch)) : 
            if update_stop : 
                break

            model.train()
            for _ in range(15) : 

                optimizer.zero_grad()
                train_nll_loss = nll_loss(F.softmax(model.last_layer1(train_X) + model.last_layer2(model.fc2(train_age_diff))), train_class)
                train_pnll_loss = pnll_loss(F.softmax(model.last_layer1(train_X) + model.last_layer2(model.fc2(train_age_diff))), train_class)

                train_nll_list[fold].append(train_nll_loss.item())
                train_pnll_list[fold].append(train_pnll_loss.item())

                train_pnll_loss.backward()
                optimizer.step()


            model.eval()
            val_nll_loss = nll_loss(F.softmax(model.last_layer1(val_X) + model.last_layer2(model.fc2(val_age_diff))), val_class).item()
            val_pnll_loss = pnll_loss(F.softmax(model.last_layer1(val_X) + model.last_layer2(model.fc2(val_age_diff))), val_class).item()

            val_nll_list[fold].append(val_nll_loss)
            val_pnll_list[fold].append(val_pnll_loss)

            if val_pnll_loss < best_loss :
                best_loss = val_pnll_loss
                best_model = copy.deepcopy(model)
                best_optim = copy.deepcopy(optimizer)
                update_count = 0
            else :
                update_count += 1

            if update_count == patience :
                update_stop = True
                print(f'{fold}th fold stopped training at {epoch - patience}th epoch')


            model.eval()


            test_nll_loss  = nll_loss(F.softmax(model.last_layer1(test_X) + model.last_layer2(model.fc2(test_age_diff))), test_class)
            test_pnll_loss = pnll_loss(F.softmax(model.last_layer1(test_X) + model.last_layer2(model.fc2(test_age_diff))), test_class)
            
            test_nll_list[fold].append(test_nll_loss.item())
            test_pnll_list[fold].append(test_pnll_loss.item())


        best_model.eval()
        fold_pred = F.softmax(best_model.last_layer1(test_X) + best_model.last_layer2(best_model.fc2(test_age_diff))).detach()

        test_nll_loss = nll_loss(fold_pred, test_class).item()
        test_pnll_loss = pnll_loss(fold_pred, test_class).item()

        # fold_pred = fold_pred.cpu()

        fold_decision = make_decision(fold_pred, K = K).int()
        fold_acc = multiclass_accuracy(fold_decision, test_class)
        print(f'{fold}-th fold test NLL, PNLL, accuracy : {test_nll_loss:.4f}, {test_pnll_loss:.4f}, {fold_acc:.4f}')

        acc_list.append(fold_acc)

        test_decision[test_indice[fold]] = fold_decision.cpu()
        test_prediction[test_indice[fold]] = fold_pred.cpu()

        # Checkpoint 1
        checkpoint = {
            'model': best_model.state_dict(),
            'optimizer': best_optim.state_dict() 
        }

        os.makedirs(f'./{experiment_name}', exist_ok=True)
        torch.save(checkpoint, f'./{experiment_name}/PNLL_age_{fold}_pretrain.pth')


    nll_loss = proposed_loss(torch.device('cpu'), K, weight)
    pnll_loss = proposed_loss(torch.device('cpu'), K, weight, p_star, eps)
    total_nll = nll_loss(torch.as_tensor(test_prediction), full_dataset.tensors[1]).item()
    total_pnll = pnll_loss(torch.as_tensor(test_prediction), full_dataset.tensors[1]).item()
    total_acc = multiclass_accuracy(test_decision, full_dataset.tensors[1])

    print(f'Total NLL, PNLL, accuracy : {total_nll:.4f}, {total_pnll:.4f}, {total_acc:.4f}')

    np.save(f'./{experiment_name}/PNLL_age_prediction', test_prediction)
    np.save(f'./{experiment_name}/PNLL_age_decision', test_decision)
    np.save(f'./{experiment_name}/PNLL_age_accuracy', acc_list)
    
    np.save(f'./{experiment_name}/PNLL_age_train_nll', train_nll_list)
    np.save(f'./{experiment_name}/PNLL_age_train_pnll', train_pnll_list)
    np.save(f'./{experiment_name}/PNLL_age_val_nll', val_nll_list)
    np.save(f'./{experiment_name}/PNLL_age_val_pnll', val_pnll_list)
    np.save(f'./{experiment_name}/PNLL_age_test_nll', test_nll_list)
    np.save(f'./{experiment_name}/PNLL_age_test_pnll', test_pnll_list)

    return [test_prediction, 
            train_nll_list, train_pnll_list, 
            val_nll_list, val_pnll_list, 
            test_nll_list, test_pnll_list]





def pnll_age_train_from_nllage_1(
    full_dataset, train_indice, val_indice, test_indice, DEVICE, SEED, 
    load_dir = './',
    N = 150, K = 3, num_fold = 10, 
    h_dim = 50, num_layers = 2, num_epoch = 200, batch_size = 8,
    weight = None, p_star = None, eps = 1e-8,
    learning_rate = 1e-3, weight_decay = 1e-2, patience = 10, experiment_name='exp') : 
    
    test_prediction = torch.zeros([N, K])
    test_decision = torch.zeros(N, dtype=torch.int32)

    acc_list = []

    train_nll_list = [[] for _ in range(num_fold)]
    train_pnll_list = [[] for _ in range(num_fold)]

    val_nll_list = [[] for _ in range(num_fold)]
    val_pnll_list = [[] for _ in range(num_fold)]

    test_nll_list = [[] for _ in range(num_fold)]
    test_pnll_list = [[] for _ in range(num_fold)]

    nll_loss = proposed_loss(DEVICE, K, weight)
    pnll_loss = proposed_loss(DEVICE, K, weight, p_star, eps)
    
    for fold in range(num_fold) : 
        print(f"{fold}th fold starting.")
        make_reproducibility(SEED + fold)

        model = classifier(h_dim, num_layers, K).to(DEVICE)
        optimizer = optim.Adam(list(model.last_layer1.parameters()) + 
                               list(model.fc2.parameters()) + 
                               list(model.last_layer2.parameters()), lr=learning_rate, weight_decay = weight_decay)

        # train_dataset = torch.utils.data.Subset(full_dataset, indices=train_indice[fold])
        # val_dataset = torch.utils.data.Subset(full_dataset, indices=val_indice[fold])
        # test_dataset = torch.utils.data.Subset(full_dataset, indices=test_indice[fold])
        # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        # val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=15, shuffle=False)
        # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=15, shuffle=False)

        # torch_dict = torch.load(f'{load_dir}/NLL_{fold}_pretrain.pth')
        # model.load_state_dict(torch_dict['model'])
        model.load_state_dict(torch.load(f'{load_dir}/NLL_age_{fold}_pretrain.pth')['model'])

        model.eval()
        with torch.no_grad() :  
            train_X = model.get_last_hidden_layer(full_dataset.tensors[0][train_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[2][train_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[3][train_indice[fold]].to(DEVICE)).detach()
            val_X = model.get_last_hidden_layer(full_dataset.tensors[0][val_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[2][val_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[3][val_indice[fold]].to(DEVICE)).detach()
            test_X =  model.get_last_hidden_layer(full_dataset.tensors[0][test_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[2][test_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[3][test_indice[fold]].to(DEVICE)).detach()
        train_class = full_dataset.tensors[1][train_indice[fold]].to(DEVICE)
        val_class = full_dataset.tensors[1][val_indice[fold]].to(DEVICE)
        test_class = full_dataset.tensors[1][test_indice[fold]].to(DEVICE)
        
        train_age_diff = full_dataset.tensors[4][train_indice[fold]].to(DEVICE).unsqueeze(1)
        val_age_diff = full_dataset.tensors[4][val_indice[fold]].to(DEVICE).unsqueeze(1)
        test_age_diff = full_dataset.tensors[4][test_indice[fold]].to(DEVICE).unsqueeze(1)
        
        best_loss = 1e8
        best_model = copy.deepcopy(model)
        best_optim = copy.deepcopy(optimizer)
        update_count = 0
        update_stop = False

        for epoch in tqdm(range(num_epoch)) : 
            if update_stop : 
                break

            model.train()
            for _ in range(15) : 

                optimizer.zero_grad()
                train_nll_loss = nll_loss(F.softmax(model.last_layer1(train_X) + model.last_layer2(model.fc2(train_age_diff))), train_class)
                train_pnll_loss = pnll_loss(F.softmax(model.last_layer1(train_X) + model.last_layer2(model.fc2(train_age_diff))), train_class)

                train_nll_list[fold].append(train_nll_loss.item())
                train_pnll_list[fold].append(train_pnll_loss.item())

                train_pnll_loss.backward()
                optimizer.step()


            model.eval()
            val_nll_loss = nll_loss(F.softmax(model.last_layer1(val_X) + model.last_layer2(model.fc2(val_age_diff))), val_class).item()
            val_pnll_loss = pnll_loss(F.softmax(model.last_layer1(val_X) + model.last_layer2(model.fc2(val_age_diff))), val_class).item()

            val_nll_list[fold].append(val_nll_loss)
            val_pnll_list[fold].append(val_pnll_loss)

            if val_pnll_loss < best_loss :
                best_loss = val_pnll_loss
                best_model = copy.deepcopy(model)
                best_optim = copy.deepcopy(optimizer)
                update_count = 0
            else :
                update_count += 1

            if update_count == patience :
                update_stop = True
                print(f'{fold}th fold stopped training at {epoch - patience}th epoch')


            model.eval()


            test_nll_loss  = nll_loss(F.softmax(model.last_layer1(test_X) + model.last_layer2(model.fc2(test_age_diff))), test_class)
            test_pnll_loss = pnll_loss(F.softmax(model.last_layer1(test_X) + model.last_layer2(model.fc2(test_age_diff))), test_class)
            
            test_nll_list[fold].append(test_nll_loss.item())
            test_pnll_list[fold].append(test_pnll_loss.item())


        best_model.eval()
        fold_pred = F.softmax(best_model.last_layer1(test_X) + best_model.last_layer2(best_model.fc2(test_age_diff))).detach()

        test_nll_loss = nll_loss(fold_pred, test_class).item()
        test_pnll_loss = pnll_loss(fold_pred, test_class).item()

        # fold_pred = fold_pred.cpu()

        fold_decision = make_decision(fold_pred, K = K).int()
        fold_acc = multiclass_accuracy(fold_decision, test_class)
        print(f'{fold}-th fold test NLL, PNLL, accuracy : {test_nll_loss:.4f}, {test_pnll_loss:.4f}, {fold_acc:.4f}')

        acc_list.append(fold_acc)

        test_decision[test_indice[fold]] = fold_decision.cpu()
        test_prediction[test_indice[fold]] = fold_pred.cpu()

        # Checkpoint 1
        checkpoint = {
            'model': best_model.state_dict(),
            'optimizer': best_optim.state_dict() 
        }

        os.makedirs(f'./{experiment_name}', exist_ok=True)
        torch.save(checkpoint, f'./{experiment_name}/PNLL_age_{fold}_pretrain.pth')


    nll_loss = proposed_loss(torch.device('cpu'), K, weight)
    pnll_loss = proposed_loss(torch.device('cpu'), K, weight, p_star, eps)
    total_nll = nll_loss(torch.as_tensor(test_prediction), full_dataset.tensors[1]).item()
    total_pnll = pnll_loss(torch.as_tensor(test_prediction), full_dataset.tensors[1]).item()
    total_acc = multiclass_accuracy(test_decision, full_dataset.tensors[1])

    print(f'Total NLL, PNLL, accuracy : {total_nll:.4f}, {total_pnll:.4f}, {total_acc:.4f}')

    np.save(f'./{experiment_name}/PNLL_age_prediction', test_prediction)
    np.save(f'./{experiment_name}/PNLL_age_decision', test_decision)
    np.save(f'./{experiment_name}/PNLL_age_accuracy', acc_list)
    
    np.save(f'./{experiment_name}/PNLL_age_train_nll', train_nll_list)
    np.save(f'./{experiment_name}/PNLL_age_train_pnll', train_pnll_list)
    np.save(f'./{experiment_name}/PNLL_age_val_nll', val_nll_list)
    np.save(f'./{experiment_name}/PNLL_age_val_pnll', val_pnll_list)
    np.save(f'./{experiment_name}/PNLL_age_test_nll', test_nll_list)
    np.save(f'./{experiment_name}/PNLL_age_test_pnll', test_pnll_list)

    return [test_prediction, 
            train_nll_list, train_pnll_list, 
            val_nll_list, val_pnll_list, 
            test_nll_list, test_pnll_list]





def pnll_age_train_from_nllage_2(
    full_dataset, train_indice, val_indice, test_indice, DEVICE, SEED, 
    load_dir = './',
    N = 150, K = 3, num_fold = 10, 
    h_dim = 50, num_layers = 2, num_epoch = 200, batch_size = 8,
    weight = None, p_star = None, eps = 1e-8,
    learning_rate = 1e-3, weight_decay = 1e-2, patience = 10, experiment_name='exp') : 
    
    test_prediction = torch.zeros([N, K])
    test_decision = torch.zeros(N, dtype=torch.int32)

    acc_list = []

    train_nll_list = [[] for _ in range(num_fold)]
    train_pnll_list = [[] for _ in range(num_fold)]

    val_nll_list = [[] for _ in range(num_fold)]
    val_pnll_list = [[] for _ in range(num_fold)]

    test_nll_list = [[] for _ in range(num_fold)]
    test_pnll_list = [[] for _ in range(num_fold)]

    nll_loss = proposed_loss(DEVICE, K, weight)
    pnll_loss = proposed_loss(DEVICE, K, weight, p_star, eps)
    
    for fold in range(num_fold) : 
        print(f"{fold}th fold starting.")
        make_reproducibility(SEED + fold)

        model = classifier(h_dim, num_layers, K).to(DEVICE)
        optimizer = optim.Adam(list(model.last_layer1.parameters()) + 
                               list(model.last_layer2.parameters()), lr=learning_rate, weight_decay = weight_decay)

        # train_dataset = torch.utils.data.Subset(full_dataset, indices=train_indice[fold])
        # val_dataset = torch.utils.data.Subset(full_dataset, indices=val_indice[fold])
        # test_dataset = torch.utils.data.Subset(full_dataset, indices=test_indice[fold])
        # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        # val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=15, shuffle=False)
        # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=15, shuffle=False)

        # torch_dict = torch.load(f'{load_dir}/NLL_{fold}_pretrain.pth')
        # model.load_state_dict(torch_dict['model'])
        model.load_state_dict(torch.load(f'{load_dir}/NLL_age_{fold}_pretrain.pth')['model'])

        model.eval()
        with torch.no_grad() :  
            train_X = model.get_last_hidden_layer(full_dataset.tensors[0][train_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[2][train_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[3][train_indice[fold]].to(DEVICE)).detach()
            val_X = model.get_last_hidden_layer(full_dataset.tensors[0][val_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[2][val_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[3][val_indice[fold]].to(DEVICE)).detach()
            test_X =  model.get_last_hidden_layer(full_dataset.tensors[0][test_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[2][test_indice[fold]].to(DEVICE), 
                            full_dataset.tensors[3][test_indice[fold]].to(DEVICE)).detach()
        train_class = full_dataset.tensors[1][train_indice[fold]].to(DEVICE)
        val_class = full_dataset.tensors[1][val_indice[fold]].to(DEVICE)
        test_class = full_dataset.tensors[1][test_indice[fold]].to(DEVICE)
        
        train_age_diff = full_dataset.tensors[4][train_indice[fold]].to(DEVICE).unsqueeze(1)
        val_age_diff = full_dataset.tensors[4][val_indice[fold]].to(DEVICE).unsqueeze(1)
        test_age_diff = full_dataset.tensors[4][test_indice[fold]].to(DEVICE).unsqueeze(1)

        train_Z = model.fc2(train_age_diff).detach()
        val_Z = model.fc2(val_age_diff).detach()
        test_Z = model.fc2(test_age_diff).detach()
        
        best_loss = 1e8
        best_model = copy.deepcopy(model)
        best_optim = copy.deepcopy(optimizer)
        update_count = 0
        update_stop = False

        for epoch in tqdm(range(num_epoch)) : 
            if update_stop : 
                break

            model.train()
            for _ in range(15) : 

                optimizer.zero_grad()
                train_nll_loss = nll_loss(F.softmax(model.last_layer1(train_X) + model.last_layer2(train_Z)), train_class)
                train_pnll_loss = pnll_loss(F.softmax(model.last_layer1(train_X) + model.last_layer2(train_Z)), train_class)

                train_nll_list[fold].append(train_nll_loss.item())
                train_pnll_list[fold].append(train_pnll_loss.item())

                train_pnll_loss.backward()
                optimizer.step()


            model.eval()
            val_nll_loss = nll_loss(F.softmax(model.last_layer1(val_X) + model.last_layer2(val_Z)), val_class).item()
            val_pnll_loss = pnll_loss(F.softmax(model.last_layer1(val_X) + model.last_layer2(val_Z)), val_class).item()

            val_nll_list[fold].append(val_nll_loss)
            val_pnll_list[fold].append(val_pnll_loss)

            if val_pnll_loss < best_loss :
                best_loss = val_pnll_loss
                best_model = copy.deepcopy(model)
                best_optim = copy.deepcopy(optimizer)
                update_count = 0
            else :
                update_count += 1

            if update_count == patience :
                update_stop = True
                print(f'{fold}th fold stopped training at {epoch - patience}th epoch')


            model.eval()


            test_nll_loss  = nll_loss(F.softmax(model.last_layer1(test_X) + model.last_layer2(test_Z)), test_class)
            test_pnll_loss = pnll_loss(F.softmax(model.last_layer1(test_X) + model.last_layer2(test_Z)), test_class)
            
            test_nll_list[fold].append(test_nll_loss.item())
            test_pnll_list[fold].append(test_pnll_loss.item())


        best_model.eval()
        fold_pred = F.softmax(best_model.last_layer1(test_X) + best_model.last_layer2(test_Z)).detach()

        test_nll_loss = nll_loss(fold_pred, test_class).item()
        test_pnll_loss = pnll_loss(fold_pred, test_class).item()

        # fold_pred = fold_pred.cpu()

        fold_decision = make_decision(fold_pred, K = K).int()
        fold_acc = multiclass_accuracy(fold_decision, test_class)
        print(f'{fold}-th fold test NLL, PNLL, accuracy : {test_nll_loss:.4f}, {test_pnll_loss:.4f}, {fold_acc:.4f}')

        acc_list.append(fold_acc)

        test_decision[test_indice[fold]] = fold_decision.cpu()
        test_prediction[test_indice[fold]] = fold_pred.cpu()

        # Checkpoint 1
        checkpoint = {
            'model': best_model.state_dict(),
            'optimizer': best_optim.state_dict() 
        }

        os.makedirs(f'./{experiment_name}', exist_ok=True)
        torch.save(checkpoint, f'./{experiment_name}/PNLL_age_{fold}_pretrain.pth')


    nll_loss = proposed_loss(torch.device('cpu'), K, weight)
    pnll_loss = proposed_loss(torch.device('cpu'), K, weight, p_star, eps)
    total_nll = nll_loss(torch.as_tensor(test_prediction), full_dataset.tensors[1]).item()
    total_pnll = pnll_loss(torch.as_tensor(test_prediction), full_dataset.tensors[1]).item()
    total_acc = multiclass_accuracy(test_decision, full_dataset.tensors[1])

    print(f'Total NLL, PNLL, accuracy : {total_nll:.4f}, {total_pnll:.4f}, {total_acc:.4f}')

    np.save(f'./{experiment_name}/PNLL_age_prediction', test_prediction)
    np.save(f'./{experiment_name}/PNLL_age_decision', test_decision)
    np.save(f'./{experiment_name}/PNLL_age_accuracy', acc_list)
    
    np.save(f'./{experiment_name}/PNLL_age_train_nll', train_nll_list)
    np.save(f'./{experiment_name}/PNLL_age_train_pnll', train_pnll_list)
    np.save(f'./{experiment_name}/PNLL_age_val_nll', val_nll_list)
    np.save(f'./{experiment_name}/PNLL_age_val_pnll', val_pnll_list)
    np.save(f'./{experiment_name}/PNLL_age_test_nll', test_nll_list)
    np.save(f'./{experiment_name}/PNLL_age_test_pnll', test_pnll_list)

    return [test_prediction, 
            train_nll_list, train_pnll_list, 
            val_nll_list, val_pnll_list, 
            test_nll_list, test_pnll_list]