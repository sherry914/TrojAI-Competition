import torch
import os
import helper

# Retrieve the sorted list of filenames from the root directory.
fnames = sorted(os.listdir(helper.root()))

ims = []
gts = []
# Initialize lists to store unique images and their corresponding ground truths.
# Using dictionaries to store seen images and ground truths by their identifiers if available.
seen_images = {}

for fname in fnames:
    # Create an engine instance for each folder within the root directory.
    folder_path = os.path.join(helper.root(), fname)
    interface = helper.engine(folder=folder_path)
    
    # Load example data using the engine instance.
    try:
        data = interface.load_examples()
        if data is not None and data['im'] not in seen_images:
            seen_images[data['im']] = True
            ims.append(data['im'])
            gts.append(data['gt'])
    except Exception as e:
        print(f"Failed to load or process data from {folder_path}: {e}")

# Convert lists of tensors into single tensors for images and ground truths.
ims = torch.cat(ims, dim=0)
gts = torch.cat(gts, dim=0)

# Save the concatenated images and ground truths as a single file.
data_path = './learned_parameters/data_samples.pt'
torch.save({'im': ims, 'gt': gts}, data_path)

# Load the saved data for further processing or analysis.
try:
    data = torch.load(data_path)
    print(data['im'].shape, data['gt'].shape)
except Exception as e:
    print(f"Failed to load data from {data_path}: {e}")

# # Initialize an engine instance for a specific folder to demonstrate usage of `get_logits`.
# interface = helper.engine(folder=os.path.join(helper.root(), 'id-00000048'))

# # Get logits from the interface using the loaded data.
# try:
#     s = interface.get_logits(data)
#     print(s.shape)
# except AttributeError:
#     print("Error: `get_logits` method is not implemented in the `helper.engine` class.")
