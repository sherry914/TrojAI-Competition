from typing import Dict
from pathlib import Path

import torchvision
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from amp import AMP
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn

from trojai_mitigation_round.mitigations.mitigation import TrojAIMitigation
from trojai_mitigation_round.mitigations.mitigated_model import TrojAIMitigatedModel

def self_test_model(model, poi_set, clean_set, batch_size, num_workers, device):

    dataloader = torch.utils.data.DataLoader(clean_set, batch_size=batch_size, num_workers=num_workers)
    
    model.eval()

    correct_total = 0
    num_samples = 0

    with torch.no_grad():
        for x, y, fname in tqdm(dataloader):
            # Move your inputs to the device
            preprocess_x = x.to(device)
            
            # Forward pass
            output_logits = model(preprocess_x).cpu()
            
            # If y is not None, compute accuracy
            if y is not None:
                # Get the predicted class for each sample
                predictions = output_logits.argmax(dim=1)
                
                # Count how many predictions match the ground truth
                correct_total += (predictions == y).sum().item()
                num_samples += y.size(0)

    # Compute accuracy only if labels exist (num_samples > 0)
    if num_samples > 0:
        accuracy = correct_total / num_samples
        print(f"Accuracy: {accuracy:.4f}")
    else:
        print("No labels available to compute accuracy.")

    if poi_set:
        dataloader = torch.utils.data.DataLoader(poi_set, batch_size=batch_size, num_workers=num_workers)
        
        model.eval()

        correct_total = 0
        num_samples = 0

        with torch.no_grad():
            for x, y, fname in tqdm(dataloader):
                # Move your inputs to the device
                preprocess_x = x.to(device)
                
                # Forward pass
                output_logits = model(preprocess_x).cpu()
                
                # If y is not None, compute accuracy
                if y is not None:
                    # Get the predicted class for each sample
                    predictions = output_logits.argmax(dim=1)
                    
                    # Count how many predictions match the ground truth
                    correct_total += (predictions == y).sum().item()
                    num_samples += y.size(0)

        # Compute accuracy only if labels exist (num_samples > 0)
        if num_samples > 0:
            accuracy = correct_total / num_samples
            print(f"ASR: {accuracy:.4f}")
        else:
            print("No labels available to compute accuracy.")

class Adapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(Adapter, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Simple residual connection: output = x + AdapterBlock(x)
        return x + self.fc2(self.activation(self.fc1(x)))


def reset_head(model: nn.Module):

    # torchvision.models.vision_transformer.VisionTransformer 的 head 通常是 nn.Linear
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        model.head.reset_parameters()


class FineTuningTrojai(TrojAIMitigation):
    def __init__(self, device, loss_cls, optim_cls, lr, epochs, ckpt_dir="./ckpts", ckpt_every=0, batch_size=32, num_workers=1, **kwargs):
        super().__init__(device, batch_size, num_workers, **kwargs)
        self._optimizer_class = optim_cls
        self._loss_cls = loss_cls
        self.lr = lr
        self.epochs = epochs
        self.ckpt_dir = ckpt_dir
        self.ckpt_every = ckpt_every


    def mitigate_model(self, model: torch.nn.Module, dataset: Dataset) -> TrojAIMitigatedModel:
        model = model.to(self.device)
        model.train()
        optimizer = self._optimizer_class(model.parameters(), lr=self.lr)
        loss_fn = self._loss_cls()
        trainloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=False
        )

        return TrojAIMitigatedModel(model)

    def mitigate_model_online(self, model: torch.nn.Module, dataset: Dataset) -> TrojAIMitigatedModel:

        # clean_set = torch.load("clean_set_0.pth")
        # poi_set = None
        # poi_set = torch.load("poi_set.pth")

        factor = 4

        model_name = model.__class__.__name__
        print(model_name)

        if "VisionTransformer" in model_name:
            lr = 0.001
            # eps = 0.2/factor
            # eps = 1.0/factor
            eps = 0.05
            epoch_all = 15
        elif "MobileNetV2" in model_name:
            lr = 0.1
            eps = 2.0/factor
            epoch_all = 30
        else:
            lr = 1.0
            eps = 2.0/factor
            epoch_all = 30



        # optimizer = optim.SGD(
        #     model.parameters(),
        #     lr=0.01,
        #     momentum=0.9,
        #     weight_decay=5e-4
        # )

        def freeze_except_linear(model: nn.Module):
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    for param in module.parameters():
                        param.requires_grad = True
                else:
                    for param in module.parameters():
                        param.requires_grad = False

        def reset_linear_layers(model):
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    module.reset_parameters()

        def freeze_vit_backbone_unfreeze_head(model: nn.Module):

            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True

        def freeze_all_except_bias_ln_and_header(model: nn.Module):

            for param in model.parameters():
                param.requires_grad = False

            for name, param in model.named_parameters():
                if name.endswith(".bias"):
                    param.requires_grad = True

            for module in model.modules():
                if isinstance(module, nn.LayerNorm):
                    for p in module.parameters():
                        p.requires_grad = True

            for param in model.head.parameters():
                param.requires_grad = True

        def reset_vit_head(model: nn.Module):
            if isinstance(model.head, nn.Linear):
                model.head.reset_parameters()
            else:
                for layer in model.head.modules():
                    if isinstance(layer, nn.Linear):
                        layer.reset_parameters()

        for _ in range(1):

            if "VisionTransformer" in model_name:
                # reset_vit_head(model)




                # for name, param in model.named_parameters():
                #     param.requires_grad = False

                # num_classes = model.num_classes
                # in_features = model.head.in_features
                # out_features = model.head.out_features

                # model.head = nn.Sequential(
                #     Adapter(in_features, hidden_dim=64),
                #     nn.Linear(in_features, out_features)
                # )




                replace_attn_with_lora(model, r=4, alpha=4)
                reset_head(model)
                # freeze_backbone_except_lora_and_head(model)



                # freeze_all_except_bias_ln_and_header(model)
                # freeze_vit_backbone_unfreeze_head(model)
            else:
                reset_linear_layers(model)
                # freeze_except_linear(model)

            model = model.to(self.device)


            # noise_scale = 0.05 
            # with torch.no_grad():
            #     for param in model.parameters():
            #         param.data += noise_scale * torch.randn_like(param)

            if "VisionTransformer" in model_name:
                optimizer = AMP(params=filter(lambda p: p.requires_grad, model.parameters()),
                                lr=lr,
                                epsilon=eps,
                                inner_lr=eps*2,
                                inner_iter=1,
                                base_optimizer=optim.AdamW)
            else:
                optimizer = AMP(params=filter(lambda p: p.requires_grad, model.parameters()),
                                lr=lr,
                                epsilon=eps,
                                inner_lr=eps*2,
                                inner_iter=1,
                                base_optimizer=optim.SGD,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=True)

            # scheduler = MultiStepLR(optimizer, milestones=[10], gamma=0.3)

            loss_fn = self._loss_cls()
            trainloader = DataLoader(
                dataset,
                batch_size=256,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=False,
                pin_memory=False
            )

            # self_test_model(model, poi_set, clean_set, self.batch_size, self.num_workers, self.device)
            # if "VisionTransformer" not in model_name:
            if 1:
                for epoch in range(epoch_all):
                    model.train()
                    for step, (x, y, fname) in enumerate(trainloader):
                        x = x.to(self.device)
                        y = y.to(self.device)

                        # optimizer.zero_grad()
                        # pred = model(x)
                        # loss = loss_fn(pred, y)
                        # loss.backward()
                        # optimizer.step()

                        def closure():
                            optimizer.zero_grad()
                            log_probs = model(x)
                            loss = loss_fn(log_probs, y)
                            loss.backward()
                            return log_probs, loss

                        log_probs, loss = optimizer.step(closure)
                    
                    print(f"Epoch {epoch} end, loss: {loss:.4f}")
                    # scheduler.step()

                    # if (epoch+1) % 5 == 0:
                    #     self_test_model(model, poi_set, clean_set, self.batch_size, self.num_workers, self.device)

        return TrojAIMitigatedModel(model)

