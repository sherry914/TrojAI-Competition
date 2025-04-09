import os
import time
from pathlib import Path
import importlib
import math
import torch
import torch.nn.functional as F
from arch_module import clf
from inception_module import InceptionBlock, Flatten
from torch import nn
import helper as helper
import numpy as np
import copy

epochs = 500
batch_size = 12

def to_one_hot(labels, n_classes=2, device='cuda'):
    one_hot = torch.zeros(labels.size(0), n_classes, device=device)
    return one_hot.scatter_(1, labels.long().to(device), 1)

def quantile(x,nbins):
    x=x.view(-1)
    v,_=x.sort(dim=0)
    ind=torch.arange(0,nbins).float()/(nbins-1)*(len(x)-1)
    ind=ind.to(x.device)
    ind0=torch.floor(ind).long().clamp(0,len(x)-1)
    ind1=torch.ceil(ind).long().clamp(0,len(x)-1)
    delta=(ind1-ind0).float()+1e-20
    y0=v[ind0]
    y1=v[ind1]
    y=y0+(y1-y0)*(ind-ind0)/delta
    return y


def analyze_tensor(w,bins=100):
    if w is None:
        return torch.Tensor(bins*2).fill_(0)
    else:
        hw=quantile(w,bins).cpu()
        hw_abs=quantile(w.abs(),bins).cpu()
        u,s,v=torch.svd(w)
        hs=quantile(s,bins).cpu()
        hs_abs=quantile(s.abs(),bins).cpu()
        fv=torch.cat((hw,hw_abs,hs,hs_abs),dim=0);
        return fv;

def weight2fv(g):
    fvs=[]
    for name, w in g:
        if "downsample" not in name:
            continue
        # print(name)
        if len(w.shape)==1:
            fvs.append(analyze_tensor(w.data.view(-1,1).cuda()).cpu())
        elif len(w.shape)==2:
            fvs.append(analyze_tensor(w.data.cuda()).cpu())
        elif len(w.shape)==4:
            for i in range(w.shape[-2]):
                for j in range(w.shape[-1]):
                    fvs.append(analyze_tensor(w.data[:,:,i,j].cuda()).cpu())
        else:
            print(w.shape)
            a=0/0
    
    return fvs;


def get_feature(interface, additional_data='./learned_parameters/data_samples.pt'):
    """
    Load data and characterize it by processing through a neural network interface.
    
    Parameters:
    - interface: The neural network interface for evaluating activations.
    - additional_data: The filename of the dataset to be loaded.
    
    Returns:
    A dictionary containing the sorted feature vectors ('fvs') extracted from the data.
    """
    # Attempt to load the additional data, falling back to an absolute path if necessary
    try:
        data = torch.load(additional_data)
    except:
        data = torch.load(os.path.join('/', additional_data))

    # Split data into two halves
    data_length = len(data['gt'])
    half_length = data_length // 2
    # Ensure consistent split
    torch.manual_seed(0)  # Setting the seed for reproducibility
    indices = torch.randperm(data_length)
    training_indices, evaluation_indices = indices[:half_length], indices[half_length:]
    # training_indices = torch.arange(0, half_length)
    # evaluation_indices = torch.arange(half_length, data_length)

    training_data_ori = {'im': data['im'][training_indices], 'gt': data['gt'][training_indices]}
    training_data_mis = {'im': data['im'][training_indices], 'gt': data['gt'][training_indices]}
    evaluation_data = {'im': data['im'][evaluation_indices], 'gt': data['gt'][evaluation_indices]}

    # Invert labels for training data
    training_data_mis['gt'] = 1 - training_data_mis['gt']

    # for name, param in interface.model.model.named_parameters():
    #     print(name)
    fvs_1 = weight2fv(interface.model.model.named_parameters())

    # training_images = training_data_ori['im'].cuda()
    # training_labels = training_data_ori['gt'].cuda()
    # training_labels_one_hot = to_one_hot(training_labels.unsqueeze(-1), n_classes=2, device='cuda')

    # optimizer = torch.optim.SGD(interface.model.model.parameters(), lr=1E-12, momentum=0.9)
    
    # interface.model.model.train()
    # optimizer.zero_grad()
    
    # # Forward pass
    # training_predictions = interface.model.model(training_images)
    # loss = F.binary_cross_entropy_with_logits(training_predictions, training_labels_one_hot)
    
    # # Backward pass and optimize
    # loss.backward()
    # optimizer.step()

    # interface.model.model.eval()

    # fvs_2 = weight2fv(interface.model.model.named_parameters())

    # fvs = fvs_1[2::3] + fvs_2[2::3]

    fvs = fvs_1[:3] + fvs_1[-3:]

    fvs = torch.stack(fvs,dim=0)
    print(fvs.shape)
    return {'fvs':fvs}


def weight2fv_conv(g):
    fvs=[]
    for name, w in g:
        # if "layer" not in name or "conv" not in name:
        #     continue
        if name != "layer4.1.conv1.weight" and name != "layer1.0.conv1.weight":
            continue
        # print(name)
        if len(w.shape)==1:
            fvs.append(analyze_tensor(w.data.view(-1,1).cuda()).cpu())
        elif len(w.shape)==2:
            fvs.append(analyze_tensor(w.data.cuda()).cpu())
        elif len(w.shape)==4:
            for i in range(w.shape[-2]):
                for j in range(w.shape[-1]):
                    fvs.append(analyze_tensor(w.data[:,:,i,j].cuda()).cpu())
        else:
            print(w.shape)
            a=0/0
    
    return fvs;


def get_feature_conv(interface, additional_data='./learned_parameters/data_samples.pt'):

    # for name, param in interface.model.model.named_parameters():
    #     print(name)
    # exit(0)

    fvs = weight2fv_conv(interface.model.model.named_parameters())
    # fvs = fvs[:9] + fvs[-9:]

    fvs = torch.stack(fvs,dim=0)
    print(fvs.shape)
    # exit(0)
    return {'fvs':fvs}

def weight2fv_conv2(g):
    fvs=[]
    for name, w in g:
        # if "layer" not in name or "conv" not in name:
        #     continue
        if name != "conv1.weight":
            continue
        # print(name)
        if len(w.shape)==1:
            fvs.append(analyze_tensor(w.data.view(-1,1).cuda()).cpu())
        elif len(w.shape)==2:
            fvs.append(analyze_tensor(w.data.cuda()).cpu())
        elif len(w.shape)==4:
            for i in range(w.shape[-2]):
                for j in range(w.shape[-1]):
                    fvs.append(analyze_tensor(w.data[:,:,i,j].cuda()).cpu())
        else:
            print(w.shape)
            a=0/0
    
    return fvs;


def get_feature_conv2(interface, additional_data='./learned_parameters/data_samples.pt'):

    # for name, param in interface.model.model.named_parameters():
    #     print(name)
    # exit(0)

    fvs = weight2fv_conv2(interface.model.model.named_parameters())
    # fvs = fvs[:9] + fvs[-9:]

    fvs = torch.stack(fvs,dim=0)
    print(fvs.shape)
    # exit(0)
    return {'fvs':fvs}


def get_feature_out(interface, additional_data='./learned_parameters/data_samples.pt'):
    """
    Load data and characterize it by processing through a neural network interface.
    
    Parameters:
    - interface: The neural network interface for evaluating activations.
    - additional_data: The filename of the dataset to be loaded.
    
    Returns:
    A dictionary containing the sorted feature vectors ('fvs') extracted from the data.
    """
    # Attempt to load the additional data, falling back to an absolute path if necessary
    try:
        data = torch.load(additional_data)
    except:
        data = torch.load(os.path.join('/', additional_data))

    # Split data into two halves
    data_length = 288 # len(data['gt'])
    half_length = data_length // 2
    # Ensure consistent split
    torch.manual_seed(0)  # Setting the seed for reproducibility
    indices = torch.randperm(data_length)
    training_indices, evaluation_indices = indices[:half_length], indices[half_length:]
    # training_indices = torch.arange(0, half_length)
    # evaluation_indices = torch.arange(half_length, data_length)

    training_data_ori = {'im': data['im'][training_indices], 'gt': data['gt'][training_indices]}
    training_data_mis = {'im': data['im'][training_indices], 'gt': data['gt'][training_indices]}
    evaluation_data = {'im': data['im'][evaluation_indices], 'gt': data['gt'][evaluation_indices]}

    # Invert labels for training data
    training_data_mis['gt'] = 1 - training_data_mis['gt']

    retry_times = 2
    # Process the data through the neural network to obtain feature vectors
    fvs = interface.get_logits(evaluation_data, training_data_ori, training_data_mis, retry_times)
    fvs = torch.cat(fvs, dim=0)
    fvs = F.log_softmax(fvs, dim=-1)
    gt = evaluation_data['gt'].repeat(retry_times)
    fvs = fvs.gather(1, gt.view(-1, 1).cuda()).view(-1)
    fvs = fvs.view(retry_times, half_length)

    # Sort the feature vectors
    # fvs, _ = torch.sort(fvs)
    print(fvs.shape)
    return {'fvs': fvs.data.cpu()}

def get_feature_grad(interface, additional_data='./learned_parameters/data_samples.pt'):
    try:
        data = torch.load(additional_data)
    except:
        data = torch.load(os.path.join('/', additional_data))

    training_indices = torch.arange(0, 18)
    data = {'im': data['im'][training_indices], 'gt': data['gt'][training_indices]}

    fvs = interface.get_grad(data)
    # fvs = torch.cat(fvs, dim=0)
    fvs = fvs.view(fvs.size(0), fvs.size(1) * fvs.size(2) * fvs.size(3))
    print(fvs.shape)
    return {'fvs': fvs.data.cpu()}

def build_dataset(models_dirpath):
    """
    Processes each model in a directory, extracting feature vectors and metadata.
    
    Parameters:
    - models_dirpath: Path to the directory containing the models.
    
    Returns:
    A list of dictionaries, each containing data about a model and its feature vectors.
    """
    models = sorted(os.listdir(models_dirpath))
    models = [(i, x) for i, x in enumerate(models)]

    dataset = []
    output_dir = 'conv2'  # Directory where output files will be saved
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i, fname in models:
        folder = os.path.join(models_dirpath, fname)
        interface = helper.engine(folder=folder)  # Load the model as an interface
        fvs = get_feature_conv2(interface)  # Extract feature vectors from the model

        # Load ground truth from a CSV file within the model's folder
        fname_gt = os.path.join(folder, 'ground_truth.csv')
        with open(fname_gt, 'r') as f:
            label = int(f.readline().rstrip('\n').rstrip('\r'))

        data = {'model_name': fname, 'model_id': i, 'label': label}
        fvs.update(data)

        # Save the extracted data to disk
        torch.save(fvs, f'{output_dir}/{i}.pt')
        dataset.append(fvs)
        print(f'model: {fname}')

    return dataset

def predict(ensemble, fvs, input_dim=2):
    """
    Predicts the probability that the given feature vectors belong to a certain class,
    using an ensemble of models.
    
    Parameters:
    - ensemble: A list of models in the ensemble.
    - fvs: Feature vectors to be classified.
    
    Returns:
    The predicted probability as a float.
    """
    scores = []
    with torch.no_grad():
        for i in range(len(ensemble)):
            # net = clf()
            filters = input_dim
            net =  nn.Sequential(
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
            checkpoint = ensemble[i]
            net.load_state_dict(checkpoint["net"])
            net = net.cuda()
            net.eval()

            # Calculate the log probability and adjust by temperature
            # pred = net.ano(fvs).data.cpu()
            X_val = np.array(fvs[0])
            X_val = X_val.reshape((1, X_val.shape[0], X_val.shape[1]))
            X_val = torch.tensor(X_val).float().cuda()
            pred = net(X_val)
            pred = pred[:,1] - pred[:,0]
            pred = pred.data.cpu()

            # pred = pred * math.exp(-checkpoint['T'])
            scores.append(pred)

    # Aggregate scores from all models and apply sigmoid to get the probability
    s = sum(scores) / len(scores)
    s = torch.sigmoid(s)
    return float(s)

if __name__ == "__main__":

    build_dataset(os.path.join(helper.root()))
