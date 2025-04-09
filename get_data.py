import torch
import torchvision
import json
import os

root_dir = '../trojai-datasets/cyber-network-c2-feb2024-train/models'

fnames = sorted(os.listdir(root_dir))

ims = []
gts = []
seen_images = {}

def load_examples(folder_path, examples_dirpath=None):
    if examples_dirpath is None:
        examples_dirpath = os.path.join(folder_path, 'clean-example-data')

    fnames = [fname for fname in os.listdir(examples_dirpath) if fname.endswith(('.png', '.PNG'))]
    fnames = sorted(fnames)
    images = []
    labels = []

    for fname in fnames:
        images.append(torchvision.io.read_image(os.path.join(examples_dirpath, fname)).float())
        fname_base = os.path.splitext(fname)[0]
        with open(os.path.join(examples_dirpath, f'{fname_base}.json'), 'r') as file:
            labels.append(json.load(file))

    images = torch.stack(images, dim=0)

    return {'im': images, 'gt': torch.tensor(labels, dtype=torch.long)}

for fname in fnames:
    folder_path = os.path.join(root_dir, fname)
    
    try:
        data = load_examples(folder_path)
        if data is not None and data['im'] not in seen_images:
            seen_images[data['im']] = True
            ims.append(data['im'])
            gts.append(data['gt'])
    except Exception as e:
        print(f"Failed to load or process data from {folder_path}: {e}")

ims = torch.cat(ims, dim=0)
gts = torch.cat(gts, dim=0)

data_path = './learned_parameters/clean_data.pt'
torch.save({'im': ims, 'gt': gts}, data_path)

try:
    data = torch.load(data_path)
    print(data['im'].shape, data['gt'].shape)
except Exception as e:
    print(f"Failed to load data from {data_path}: {e}")
