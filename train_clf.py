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
from arch_module import clf
import utils as utils


clfs_fin = []

def get_batches(data, batch_size=256, shuffle=False, full=False):

    if shuffle:
        ind = torch.randperm(len(next(iter(data.values()))))
        adata = {}
        for k in list(data.keys()):
            value = data[k]
            if isinstance(value, list):
                adata[k] = [value[i] for i in ind]
            else:
                adata[k] = value[torch.LongTensor(ind)].contiguous()
    else:
        adata = data

    n = len(next(iter(data.values())))
    for i in range(0, n, batch_size):
        r = min(i + batch_size, n)
        batch = dict([(k, adata[k][i:r]) for k in adata])
        
        lenn = len(next(iter(batch.values())))
        if full and lenn < batch_size:
            tmp = dict([(k, adata[k][: batch_size - lenn]) for k in adata])
            for key in batch:
                batch[key].extend(tmp[key])
            
        batch["label"] = torch.LongTensor(batch["label"])
        yield batch

    return

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
    train_bsz = 12
    val_bsz = 6
    clfs = []
    auc = []
    ce = []
    mistakes = []
    for split_id, split_set in enumerate(folds):
        train_set, val_set = split_set
        net = clf()
        net = net.cuda()
        opt = optim.Adam(net.parameters(), lr=0.001)

        best_ce = 2.0
        best_auc = 0.0
        best_net = copy.deepcopy(net)

        for _ in range(500):
            net.train()
            loss_total = []
            for batch in get_batches(
                train_set, train_bsz, shuffle=True, full=True
            ):
                opt.zero_grad()
                net.zero_grad()
                for k in batch:
                    d = batch[k]
                    if not isinstance(d, list):
                        batch[k] = d.cuda()

                gt = batch["label"]
                batch.pop("label")
                scores_i = net(batch["features"])

                spos = scores_i.gather(1, gt.view(-1, 1)).mean()
                sneg = torch.exp(scores_i).mean()
                loss = -(spos - sneg + 1)

                loss.backward()
                loss_total.append(float(loss))
                opt.step()

            loss_total = sum(loss_total) / len(loss_total)

            net.eval()
            scores = []
            gt = []
            for batch in get_batches(val_set, val_bsz):
                for k in batch:
                    d = batch[k]
                    if not isinstance(d, list):
                        batch[k] = d.cuda()
                gt.append(batch["label"].data.cpu())
                batch.pop("label")
                scores_i = net.ano(batch["features"])
                scores.append(scores_i.data.cpu())

            scores = torch.cat(scores, dim=0)
            gt = torch.cat(gt, dim=0)

            auc_i, ce_i = metrics(scores.tolist(), gt.tolist())
            # auc_i = sklearn.metrics.roc_auc_score(gt.numpy(), scores.numpy())

            if best_ce > ce_i:
                best_ce = ce_i
                best_net = copy.deepcopy(net)
            # if best_auc < auc_i:
            #     best_auc = auc_i
            #     best_net = copy.deepcopy(net)

        net = best_net
        
        net.eval()
        scores = []
        gt = []
        for batch in get_batches(val_set, val_bsz):
            for k in batch:
                d = batch[k]
                if not isinstance(d, list):
                    batch[k] = d.cuda()

            gt.append(batch["label"])
            batch.pop("label")
            scores_i = net.ano(batch["features"])
            scores.append(scores_i.data)

        scores = torch.cat(scores, dim=0)
        gt = torch.cat(gt, dim=0)

        T = torch.Tensor(1).fill_(0).cuda()
        T.requires_grad_()
        opt_T = optim.Adamax([T], lr=0.01)
        for _ in range(500):
            opt_T.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                scores * torch.exp(-T), gt.float()
            )
            loss.backward()
            opt_T.step()

        net.eval()
        scores = []
        gt = []
        for batch in get_batches(val_set, val_bsz):
            for k in batch:
                d = batch[k]
                if not isinstance(d, list):
                    batch[k] = d.cuda()

            gt.append(batch["label"].data.cpu())
            batch.pop("label")
            scores_i = net.ano(batch["features"])
            # print(torch.sigmoid(scores_i * torch.exp(-T)).data.cpu())
            scores.append((scores_i * torch.exp(-T)).data.cpu()) # Temperature


        scores = torch.cat(scores, dim=0)
        gt = torch.cat(gt, dim=0)

        auc_i, ce_i = metrics(scores.tolist(), gt.tolist())
        for i in range(len(gt)):
            if int(gt[i]) == 1 and float(scores[i]) <= 0:
                mistakes.append(4 * i + split_id)

        auc.append(auc_i)
        ce.append(ce_i)
        print(f"Fold: {split_id}, AUC: {auc_i:.4f}, CE: {ce_i:.4f}")

        clfs.append(
            {"net": net.cpu().state_dict(), "T": float(T.data.cpu())}
        )

    mistakes = sorted(mistakes)
    print("Mistakes: " + ",".join(["%d" % i for i in mistakes]))
    mistakes = set(mistakes)

    auc = torch.Tensor(auc)
    ce = torch.Tensor(ce)

    clfs_fin.extend(clfs)
    
    print("Got ", len(clfs_fin))

    torch.save(clfs_fin, f"./learned_parameters/model.pt")
    print(f"AUC mean: {auc.mean():.4f}, CE mean: {ce.mean():.4f}")

    return ce.mean()


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
                    data[k][i] = data[k][i].cuda()

    ce_list = []

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
        
    ce = train_clf(folds)
    ce_list.append(ce)
    
    ce_list = torch.Tensor(ce_list)
    print(len(clfs_fin))
    print(f"all ce mean: {ce_list.mean():.4f}")
