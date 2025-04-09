from __future__ import unicode_literals, division
from builtins import int
from builtins import range
from pyexpat import features

import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
import os
import sklearn.metrics
from collections import OrderedDict as OrderedDict
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from inception_module import InceptionBlock, Flatten, metrics
import utils as utils
from extractor import build_dataset
import helper as helper

epochs = 500
batch_size = 24

input_dim = 18

clfs_fin = []
mistakes_visited = set()


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = torch.tensor(data)
        self.label = torch.tensor(label)
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        feature = self.data[idx]
        label = self.label[idx]

        return feature, label

def CalcValLossAndAccuracy(model, loss_fn, val_loader, T=torch.tensor(0.0).cuda()):
    with torch.no_grad():
        Y_shuffled, Y_preds, losses = [],[],[]
        for i, (X, Y) in enumerate(val_loader):
            X = X.float().cuda()
            Y = Y.type(torch.LongTensor).cuda()
            preds = model(X)
            loss = loss_fn(preds, Y)
            losses.append(loss.item())

            Y_shuffled.extend(Y.cpu().tolist())
            preds = preds[:,1] - preds[:,0]
            preds = preds * torch.exp(-T)
            Y_preds.extend(preds.cpu().tolist())

        # print(Y_preds)
        # print(Y_shuffled)

        auc_i, ce_i = metrics(Y_preds, Y_shuffled)

        return auc_i, ce_i


def TrainModel(model, loss_fn, optimizer, train_loader, val_loader, epochs=10):

    best_ce = 2.0
    best_net = copy.deepcopy(model)

    for i in range(1, epochs+1):
        losses = []
        for _, (X, Y) in enumerate(train_loader):
            X = X.float().cuda()
            Y = Y.type(torch.LongTensor).cuda()
            Y_preds = model(X)
            # print(Y)
            # print(Y_preds)
            # exit(0)
            loss = loss_fn(Y_preds, Y)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # train_auc, train_ce = CalcValLossAndAccuracy(model, loss_fn, train_loader)
        val_auc, val_ce = CalcValLossAndAccuracy(model, loss_fn, val_loader)
        # print(f"Epoch: {i}, Val auc: {val_auc:.4f}, Val ce: {val_ce:.4f}")

        if best_ce > val_ce:
            best_ce = val_ce
            best_net = copy.deepcopy(model)
        
    return best_net
		
def train_clf(folds, scratch_dir):

    clfs = []
    auc = []
    ce = []
    
    for split_id, split_set in enumerate(folds):
        train_set, val_set = split_set
        X_train, X_val = train_set["features"], val_set["features"]
        y_train, y_val = np.array(train_set["label"]), np.array(val_set["label"])
        for i in range(len(X_train)):
            X_train[i] = X_train[i].tolist()
        for i in range(len(X_val)):
            X_val[i] = X_val[i].tolist()

        X_train, X_val = np.array(X_train), np.array(X_val)
        # X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        # X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

        X_train = X_train.transpose(1,0,2)
        X_train = X_train[:input_dim]
        X_train = X_train.transpose(1,0,2)
        X_val = X_val.transpose(1,0,2)
        X_val = X_val[:input_dim]
        X_val = X_val.transpose(1,0,2)
        # print(X_train.shape)
        # print(y_train.shape)
        # print(X_val.shape)
        # print(y_val.shape)

        train_set = CustomDataset(X_train, y_train)
        val_set = CustomDataset(X_val, y_val)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        learning_rate = 0.001

        loss_fn = nn.CrossEntropyLoss()
        loss_fn.cuda()

        filters = input_dim

        net = nn.Sequential(
                            InceptionBlock(
                                in_channels=input_dim,
                                n_filters=filters,
                                kernel_sizes=[5, 11, 23],
                                bottleneck_channels=filters,
                                use_residual=True,
                                activation=nn.ReLU()
                            ),
                            InceptionBlock(
                                in_channels=filters*4, 
                                n_filters=filters,
                                kernel_sizes=[5, 11, 23],
                                bottleneck_channels=filters,
                                use_residual=True,
                                activation=nn.ReLU()
                            ),
                            nn.AdaptiveAvgPool1d(output_size=1),
                            Flatten(out_features=filters*4*1),
                            nn.Linear(in_features=4*filters*1, out_features=2)
                )

        net.cuda()
        optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=0.00001)

        net = TrainModel(net, loss_fn, optimizer, train_loader, val_loader, epochs)

        with torch.no_grad():
            Y_shuffled, Y_preds, losses = [],[],[]
            for i, (X, Y) in enumerate(train_loader):
                X = X.float().cuda()
                Y = Y.type(torch.LongTensor).cuda()
                preds = net(X)
                loss = loss_fn(preds, Y)
                losses.append(loss.item())

                Y_shuffled.append(Y)
                preds = preds[:,1] - preds[:,0]
                Y_preds.append(preds)

        Y_preds = torch.cat(Y_preds, dim=0)
        Y_shuffled = torch.cat(Y_shuffled, dim=0)

        T = torch.Tensor(1).fill_(0).cuda()
        T.requires_grad_()
        opt_T = optim.Adamax([T], lr=0.01)
        for _ in range(500):
            opt_T.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                Y_preds * torch.exp(-T), Y_shuffled.float()
            )
            loss.backward()
            opt_T.step()

        val_auc, val_ce = CalcValLossAndAccuracy(net, loss_fn, val_loader, T)
        print(f"Fold: {split_id}, Final val auc: {val_auc:.4f}, Final val ce: {val_ce:.4f}")

        auc.append(val_auc)
        ce.append(val_ce)

        clfs.append(
            {"net": net.cpu().state_dict(), "T": float(T.data.cpu())}
        )

        # if val_auc < 0.6:
        #     break

    auc = torch.Tensor(auc)
    ce = torch.Tensor(ce)
    
    # if auc.min() >= 0.6:
    clfs_fin.extend(clfs)
    if len(clfs_fin) == len(folds):
        print("Got ", len(clfs_fin))
        torch.save(clfs_fin, f"./learned_parameters/{scratch_dir}.pt")

    print(f"AUC mean: {auc.mean():.4f}, CE mean: {ce.mean():.4f}")
    return auc


if __name__ == "__main__":

    auc_list = []

    for _ in range(4):

        import pandas as pd
        csv_filename = "top_50_inds.csv"
        df = pd.read_csv(csv_filename)
        selected_indices = df['Index'].tolist()

        data = {
            "model_id": [],
            "model_name": [],
            "features": [],
            "label": [],
        }

        scratch_dir = "conv2"

        data_by_model = [torch.load(os.path.join(scratch_dir,fname)) for fname in os.listdir(scratch_dir) if fname.endswith('.pt')]

        # for idx in selected_indices:
        #     if idx < len(data_by_model):
        #         data["model_name"].append(data_by_model[idx]['model_name'])
        #         data["model_id"].append(data_by_model[idx]['model_id'])
        #         data["features"].append(data_by_model[idx]['fvs'])
        #         data["label"].append(data_by_model[idx]['label'])

        for i in range(len(data_by_model)):
            data["model_name"].append(data_by_model[i]['model_name'])
            data["model_id"].append(data_by_model[i]['model_id'])
            data["features"].append(data_by_model[i]['fvs'])
            data["label"].append(data_by_model[i]['label'])

        for k in data:
            d = data[k]
            if not isinstance(d, list):
                data[k] = d.cuda()
        
        for k in data.keys():
            if isinstance(data[k], list):
                if len(data[k]) > 0 and torch.is_tensor(data[k][0]):
                    for i in range(len(data[k])):
                        data[k][i] = data[k][i]

        num_folds = 4

        folds = []
        ind = torch.randperm(len(next(iter(data.values())))).long()

        for i in range(num_folds):

            val_ind = torch.arange(i, len(next(iter(data.values()))), num_folds).long()
            val_ind = ind[val_ind]
            train_ind = list(set(ind.tolist()).difference(val_ind.tolist()))
            train_ind = torch.LongTensor(train_ind)

            split_train = {}
            for k in list(data.keys()):
                value = data[k]
                if isinstance(value, list):
                    split_train[k] = [value[i] for i in train_ind.tolist()]
                else:
                    split_train[k] = value[torch.LongTensor(train_ind.tolist())].contiguous()
            
            split_test = {}
            for k in list(data.keys()):
                value = data[k]
                if isinstance(value, list):
                    split_test[k] = [value[i] for i in val_ind.tolist()]
                else:
                    split_test[k] = value[torch.LongTensor(val_ind.tolist())].contiguous()

            folds.append((split_train, split_test))
            
        auc = train_clf(folds, scratch_dir)

        auc_list.append(auc)

        # if len(clfs_fin) >= num_folds:
        #     break
    
    auc_list = torch.cat(auc_list)
    auc_mean = torch.mean(auc_list.float()) 
    print("Mean AUC: ", auc_mean)
