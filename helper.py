import os
import json
import torch
import torchvision
import torch.nn.functional as F
from utils.models import load_model

def root():
    """
    Specifies the root directory containing model data.
    """
    return '../trojai-datasets/cyber-network-c2-feb2024-train/models'

def to_one_hot(labels, n_classes=2, device='cuda'):
    one_hot = torch.zeros(labels.size(0), n_classes, device=device)
    return one_hot.scatter_(1, labels.long().to(device), 1)

class engine:
    def __init__(self, folder=None):
        """
        Initializes the engine with a model loaded from a specified folder.
        """
        self.folder = folder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Attempt to load configuration from reduced-config.json; fall back to config.json if necessary.
        try:
            config_path = os.path.join(folder, 'reduced-config.json')
            with open(config_path, 'r') as file:
                self.config = json.load(file)
        except FileNotFoundError:
            config_path = os.path.join(folder, 'config.json')
            with open(config_path, 'r') as file:
                self.config = json.load(file)

        # Load model from the model.pt file within the specified folder.
        model, model_repr, model_class = load_model(os.path.join(folder,'model.pt'))
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.model.eval()

    def load_examples(self, examples_dirpath=None):
        """
        Loads example images and their corresponding labels from a directory.
        """
        if examples_dirpath is None:
            examples_dirpath = os.path.join(self.folder, 'clean-example-data')

        fnames = [fname for fname in os.listdir(examples_dirpath) if fname.endswith(('.png', '.PNG'))]
        fnames = sorted(fnames)
        images = []
        labels = []

        for fname in fnames:
            images.append(torchvision.io.read_image(os.path.join(examples_dirpath, fname)).float())
            fname_base = os.path.splitext(fname)[0]
            with open(os.path.join(examples_dirpath, f'{fname_base}.json'), 'r') as file:
                labels.append(json.load(file))

        images = torch.stack(images, dim=0).to(self.device)

        return {'im': images, 'gt': torch.tensor(labels, dtype=torch.long).to(self.device)}

    def get_logits(self, evaluation_data, training_data_ori, training_data_mis, retry_times):
        original_model_state = self.model.model.state_dict()
        merged_predictions = []

        training_images = training_data_ori['im'].cuda()
        training_labels = training_data_ori['gt'].cuda()
        training_labels_one_hot = to_one_hot(training_labels.unsqueeze(-1), n_classes=2, device='cuda')

        optimizer = torch.optim.SGD(self.model.model.parameters(), lr=3E-9, momentum=0.9)
        
        for epoch in range(retry_times):
            if epoch > 0:
                # Set model to training mode
                self.model.model.train()
                optimizer.zero_grad()
                
                # Forward pass
                training_predictions = self.model.model(training_images)
                loss = F.binary_cross_entropy_with_logits(training_predictions, training_labels_one_hot)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()

            # Switch to evaluation mode for inference
            self.model.model.eval()

            # Predicting logits for evaluation data without gradient calculation
            with torch.no_grad():
                evaluation_images = evaluation_data['im'].cuda()
                predictions = self.model.model(evaluation_images)

            merged_predictions.append(predictions)


        # self.model.model.load_state_dict(original_model_state)

        # training_images = training_data_mis['im'].cuda()
        # training_labels = training_data_mis['gt'].cuda()
        # training_labels_one_hot = to_one_hot(training_labels.unsqueeze(-1), n_classes=2, device='cuda')

        # optimizer = torch.optim.SGD(self.model.model.parameters(), lr=1E-12, momentum=0.9)
        
        # for epoch in range(retry_times-1):

        #     self.model.model.train()
        #     optimizer.zero_grad()
            
        #     # Forward pass
        #     training_predictions = self.model.model(training_images)
        #     loss = F.binary_cross_entropy_with_logits(training_predictions, training_labels_one_hot)
            
        #     # Backward pass and optimize
        #     loss.backward()
        #     optimizer.step()

        #     # Switch to evaluation mode for inference
        #     self.model.model.eval()

        #     # Predicting logits for evaluation data without gradient calculation
        #     with torch.no_grad():
        #         evaluation_images = evaluation_data['im'].cuda()
        #         predictions = self.model.model(evaluation_images)

        #     merged_predictions.append(predictions)

        return merged_predictions

    def get_grad(self,data):
        im=data['im'].cuda()
        gt=data['gt'].cuda()
        
        # im0=im.clone().requires_grad_()
        # pred=self.model.model(im0)
        # loss=F.cross_entropy(pred,1-gt)
        # loss.backward()
        # g0=im0.grad.data.clone()
        
        im1=im.clone().requires_grad_()
        pred=self.model.model(im1)
        loss=F.cross_entropy(pred,gt)
        loss.backward()
        g1=im1.grad.data.clone()
        
        return g1 # torch.cat((g0,g1),dim=-3)