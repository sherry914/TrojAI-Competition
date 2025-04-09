from __future__ import unicode_literals, division
from builtins import int
from builtins import range
from pyexpat import features
import utils as utils

import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
import os
import sklearn.metrics
from collections import OrderedDict as OrderedDict
import numpy as np
import pandas as pd
from sktime.transformations.panel.rocket import Rocket, MultiRocketMultivariate
from sklearn.linear_model import RidgeClassifierCV
from sklearn.utils.extmath import softmax

import warnings
warnings.filterwarnings("ignore")

def _make_column_names(column_count):
    return [f"var_{i}" for i in range(column_count)]

def from_3d_numpy_to_nested(X, column_names=None, cells_as_numpy=False):

    n_instances, n_columns, n_timepoints = X.shape
    array_type = X.dtype

    container = np.array if cells_as_numpy else pd.Series

    if column_names is None:
        column_names = _make_column_names(n_columns)

    else:
        if len(column_names) != n_columns:
            msg = " ".join(
                [
                    f"Input 3d Numpy array as {n_columns} columns,",
                    f"but only {len(column_names)} names supplied",
                ]
            )
            raise ValueError(msg)

    column_list = []
    for j, column in enumerate(column_names):
        nested_column = (
            pd.DataFrame(X[:, j, :])
            .apply(lambda x: [container(x, dtype=array_type)], axis=1)
            .str[0]
            .rename(column)
        )
        column_list.append(nested_column)
    df = pd.concat(column_list, axis=1)
    return df

def metrics(scores, gt):
    auc = float(
        sklearn.metrics.roc_auc_score(
            torch.LongTensor(gt).numpy(), torch.Tensor(scores).numpy()
        )
    )
    ce = float(
        F.binary_cross_entropy_with_logits(
            torch.Tensor(scores), torch.Tensor(gt)
        )
    )
    return auc, ce

def train_clf(folds):

    ce_list = []
    auc_list = []

    for split_id, split_set in enumerate(folds):

        train_set, val_set = split_set
        X_train, X_val = train_set["features"], val_set["features"]
        y_train, y_val = np.array(train_set["label"]), np.array(val_set["label"])
        for i in range(len(X_train)):
            X_train[i] = X_train[i].tolist()
        for i in range(len(X_val)):
            X_val[i] = X_val[i].tolist()

        X_train, X_val = np.array(X_train), np.array(X_val)
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
        # print(X_train.shape)
        # print(y_train.shape)
        # print(X_val.shape)
        # print(y_val.shape)

        X_train = from_3d_numpy_to_nested(X_train)
        X_val = from_3d_numpy_to_nested(X_val)

        # print(X_train.shape)
        # print(y_train.shape)
        auc_i = 0.0
        ce_i = 2.0

        rocket = MultiRocketMultivariate(num_kernels=10000, n_features_per_kernel=16, n_jobs=1)
        rocket.fit(X_train)
        X_train_transform = rocket.transform(X_train)
        X_test_transform = rocket.transform(X_val)

        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        classifier.fit(X_train_transform, y_train)

        d = classifier.decision_function(X_train_transform)
        d = torch.tensor(d)
        y_train = torch.tensor(y_train)

        T = torch.Tensor(1).fill_(0)
        T.requires_grad_()
        opt_T = optim.Adamax([T], lr=0.01)
        for _ in range(500):
            opt_T.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                d * torch.exp(-T), y_train.float()
            )
            loss.backward()
            opt_T.step()

        temp = {"T": T}

        torch.save(temp, f"./learned_parameters/T_{split_id}.pt")

        T = torch.load(f"./learned_parameters/T_{split_id}.pt")["T"]

        d = classifier.decision_function(X_test_transform)
        d = torch.tensor(d)

        # if arch == 'resnet50':
        d2 = d * torch.exp(-T)

        auc_i, ce_i = metrics(d.tolist(), y_val.tolist())
        print(split_id)

        print(auc_i, ce_i)

        auc_i, ce_i = metrics(d2.tolist(), y_val.tolist())
        print(auc_i, ce_i)

        ce_list.append(ce_i)
        auc_list.append(auc_i)

        import pickle

        with open(f'./learned_parameters/rocket_{split_id}.pkl','wb') as f:
            pickle.dump(rocket,f)

        with open(f'./learned_parameters/clf_{split_id}.pkl','wb') as f:
            pickle.dump(classifier,f)

    print("Mean CE: ", np.mean(ce_list))
    print("Mean AUC: ", np.mean(auc_list))


if __name__ == "__main__":
    data = {
        "model_id": [],
        "model_name": [],
        "features": [],
        "label": [],
    }

    data_by_model = [torch.load(os.path.join("scratch",fname)) for fname in os.listdir("scratch") if fname.endswith('.pt')]

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

    folds = []
    for i in range(4):
        ind = torch.randperm(len(next(iter(data.values())))).long()
        val_ind = torch.arange(i, len(next(iter(data.values()))), 4).long()
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
        
    train_clf(folds)

