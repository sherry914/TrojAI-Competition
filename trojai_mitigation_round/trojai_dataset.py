import json
from pathlib import Path

import torch
import torchvision
from torchvision.transforms import v2
import io
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# If you want to allow loading truncated images (optional):
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_image_stable(full_path):
    with open(full_path, 'rb') as f:
        file_data = f.read()
    
    with Image.open(io.BytesIO(file_data)) as img:
        img.load()
        
        return img.copy()


# class Round11SampleDataset(Dataset):
#     def __init__(self, root, img_exts=["jpg", "png"], class_info_ext="json", split='test', require_label=False):
#         root = Path(root)
#         train_augmentation_transforms = torchvision.transforms.Compose(
#             [
#                 v2.PILToTensor(),
#                 v2.RandomPhotometricDistort(),
#                 torchvision.transforms.RandomCrop(size=256, padding=int(10), padding_mode='reflect'),
#                 torchvision.transforms.RandomHorizontalFlip(),
#                 torchvision.transforms.ConvertImageDtype(torch.float),
#             ]
#         )

#         test_augmentation_transforms = torchvision.transforms.Compose(
#             [
#                 v2.PILToTensor(),
#                 torchvision.transforms.ConvertImageDtype(torch.float),
#             ]
#         )
#         if split == 'test':
#             self.transform = test_augmentation_transforms
#         else:
#             self.transform = train_augmentation_transforms

#         self._img_directory_contents = sorted([path for path in root.glob("*.*") if path.suffix[1:] in img_exts])

#         self.data = []
#         self.fnames = []
#         for img_fname in self._img_directory_contents:
#             full_path = root / img_fname
            
#             if require_label:
#                 json_path = root / Path(img_fname.stem + f".{class_info_ext}")
#                 assert json_path.exists(), f"No {class_info_ext} found for {img_fname}"
#                 with open(json_path, 'r') as f:
#                     label = json.load(f)
#             else:
#                 label = -1

#             pil_img = Image.open(full_path)
#             self.fnames.append(img_fname.name)
#             self.data.append((pil_img, label))


#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img, label = self.data[idx]
#         fname = self.fnames[idx]

#         img = self.transform(img)
#         return img, label, fname



class Round11SampleDataset(Dataset):
    def __init__(
        self, 
        root, 
        img_exts=["jpg", "png"], 
        class_info_ext="json", 
        split='test', 
        require_label=False
    ):
        root = Path(root)

        train_augmentation_transforms = torchvision.transforms.Compose([
            v2.PILToTensor(),
            v2.RandomPhotometricDistort(),
            torchvision.transforms.RandomCrop(size=256, padding=10, padding_mode='reflect'),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ConvertImageDtype(torch.float),
        ])

        test_augmentation_transforms = torchvision.transforms.Compose([
            v2.PILToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float),
        ])

        # Assign the appropriate transformations based on split
        if split == 'test':
            self.transform = test_augmentation_transforms
        else:
            self.transform = train_augmentation_transforms

        # Gather all image files with the allowed extensions
        self._img_directory_contents = sorted([
            path for path in root.glob("*.*") if path.suffix[1:] in img_exts
        ])

        self.data = []
        self.fnames = []

        for img_fname in self._img_directory_contents:
            filename = img_fname.name
            full_path = root / filename

            if require_label:
                json_path = root / Path(img_fname.stem + f".{class_info_ext}")
                assert json_path.exists(), f"No {class_info_ext} found for {img_fname}"

                with open(json_path, 'r') as f:
                    label_data = json.load(f)

                    # Handle different JSON formats
                    if isinstance(label_data, int):
                        # Case 1: A single integer
                        label = label_data
                    elif isinstance(label_data, dict):
                        # Case 2/3: Dictionary with clean_label / poisoned_label
                        if "poisoned_label" in label_data:
                            label = label_data["poisoned_label"]
                        else:
                            # If no poisoned_label, we expect clean_label to exist
                            assert "clean_label" in label_data, \
                                f"JSON file {json_path} missing 'clean_label'"
                            label = label_data["clean_label"]
                    else:
                        raise ValueError(
                            f"Invalid JSON format in file {json_path}. "
                            "Must be either a single int or a dict with 'clean_label'/'poisoned_label'."
                        )
            else:
                label = -1

            pil_img = load_image_stable(full_path)
            self.fnames.append(img_fname.name)
            self.data.append((pil_img, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        fname = self.fnames[idx]

        # Apply the chosen transform to the image
        img = self.transform(img)

        return img, label, fname
