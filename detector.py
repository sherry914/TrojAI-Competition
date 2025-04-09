# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import logging
import os
import json
import jsonpickle
import pickle
import numpy as np

from sklearn.ensemble import RandomForestRegressor

import utils.models
from utils.abstract import AbstractDetector
from utils.models import load_model, load_models_dirpath

import torch
import torchvision
import skimage.io

# Inference imports
from gentle.rl.opac2_multitask import OffPolicyActorCriticMultitask
from clean_envs import make_clean_env
import sys
import json
import torch
import time
import cv2
from box import Box
from gentle.common.utils import get_sampler


class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_filepath = os.path.join(self.learned_parameters_dirpath, "model.bin")
        self.models_padding_dict_filepath = os.path.join(self.learned_parameters_dirpath, "models_padding_dict.bin")
        self.model_layer_map_filepath = os.path.join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.layer_transform_filepath = os.path.join(self.learned_parameters_dirpath, "layer_transform.bin")

        self.input_features = metaparameters["train_input_features"]
        self.weight_params = {
            "rso_seed": metaparameters["train_weight_rso_seed"]
        }

    def write_metaparameters(self):
        metaparameters = {
            "train_input_features": self.input_features,
            "train_weight_rso_seed": self.weight_params["rso_seed"]
        }

        with open(os.path.join(self.learned_parameters_dirpath, os.path.basename(self.metaparameter_filepath)), "w") as fp:
            fp.write(jsonpickle.encode(metaparameters, warn=True, indent=2))

    def generate_features(self, model_dict):

        import torch
        import math

        def get_dist(w, bins=100):
            sorted_w, _ = w.sort()
            lenn = sorted_w.numel()
            indices = torch.linspace(0, lenn - 1, steps=bins).long()
            hist = sorted_w[indices]
            return hist

        def get_feature(param):
            if param.dim() == 1:
                return get_feature(param.view(1, -1))
            elif param.dim() in [2, 4]:
                w = param.view(-1)
                w_hist = get_dist(w)
                w_abs_hist = get_dist(w.abs())
                feature = torch.cat((w_hist, w_abs_hist), dim=0)
                return [feature]
            else:
                return []

        features = []
        for keyy in ['pi', 'q1', 'v']:
            model_state = model_dict[keyy]
            layer_stats = []
            for key, tensor in model_state.items():
                if key.startswith('mlp'):
                    flat_tensor = tensor.view(-1)
                    stats = torch.tensor([
                        flat_tensor.min(),
                        flat_tensor.max(),
                        flat_tensor.mean(),
                        flat_tensor.std()
                    ], device=flat_tensor.device)

                    if torch.isnan(stats).any():
                        stats = torch.nan_to_num(stats, nan=0.0)

                    # Histogram-based features
                    hist_features = get_feature(tensor)
                    if hist_features:
                        hist_features = torch.cat(hist_features)
                        stats = torch.cat([stats, hist_features])

                    # SVD features for 4D weight matrices
                    if tensor.dim() == 4 and key.endswith('.weight'):
                        reshaped_weight = tensor.view(tensor.shape[0], -1)
                        u, s, v = torch.svd(reshaped_weight)
                        num_svd_features = min(10, s.size(0))
                        svd_features = s[:num_svd_features]
                        if num_svd_features < 10:
                            padding = torch.zeros(10 - num_svd_features, device=s.device)
                            svd_features = torch.cat([svd_features, padding])
                        svd_stats = torch.tensor([
                            s.max(),
                            s.min(),
                            s.mean()
                        ], device=s.device)
                        stats = torch.cat([stats, svd_stats, svd_features])

                    layer_stats.append(stats)

            if layer_stats:
                layer_stats = torch.stack(layer_stats)  # Shape: (num_layers, num_features_per_layer)
                # Compute summary statistics over layers
                features_mean = layer_stats.mean(dim=0)
                features_max = layer_stats.max(dim=0)[0]
                features_min = layer_stats.min(dim=0)[0]
                features_std = layer_stats.std(dim=0)

                # Concatenate the summary statistics
                features.append(torch.cat([features_mean, features_max, features_min, features_std]))
            else:
                # Handle case where there are no layers
                features.append(torch.empty(0))

        if features:
            model_feats = torch.cat(features).unsqueeze(0)
        else:
            model_feats = torch.empty((1, 0))
        return model_feats


    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        for random_seed in np.random.randint(1000, 9999, 10):
            self.weight_params["rso_seed"] = random_seed
            self.manual_configure(models_dirpath)

    # def manual_configure(self, models_dirpath: str):
    #     """Configuration of the detector using the parameters from the metaparameters
    #     JSON file.

    #     Args:
    #         models_dirpath: str - Path to the list of model to use for training
    #     """
    #     # Create the learned parameter folder if needed
    #     if not os.path.exists(self.learned_parameters_dirpath):
    #         os.makedirs(self.learned_parameters_dirpath)

    #     # List all available model
    #     model_path_list = sorted([os.path.join(models_dirpath, model) for model in os.listdir(models_dirpath)])
    #     logging.info(f"Loading %d models...", len(model_path_list))

    #     model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)

    #     logging.info("Building RandomForest based on random features, with the provided mean and std.")
    #     rso = np.random.RandomState(seed=self.weight_params['rso_seed'])
    #     X = []
    #     y = []
    #     for model_arch in model_repr_dict.keys():
    #         for model_index in range(len(model_repr_dict[model_arch])):
    #             y.append(model_ground_truth_dict[model_arch][model_index])

    #             model_feats = rso.normal(loc=self.weight_params['mean'], scale=self.weight_params['std'], size=(1,self.input_features))
    #             X.append(model_feats)
    #     X = np.vstack(X)

    #     logging.info("Training RandomForestRegressor model.")
    #     model = RandomForestRegressor(**self.random_forest_kwargs, random_state=0)
    #     model.fit(X, y)

    #     logging.info("Saving RandomForestRegressor model...")
    #     with open(self.model_filepath, "wb") as fp:
    #         pickle.dump(model, fp)

    #     self.write_metaparameters()
    #     logging.info("Configuration done!")

    def manual_configure(self, models_dirpath: str):
        """Configure the detector using parameters from the metaparameters JSON file.

        Args:
            models_dirpath: str - Path to the directory containing models for training.
        """
        import os
        import logging
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        # Create the learned parameters directory if it doesn't exist
        os.makedirs(self.learned_parameters_dirpath, exist_ok=True)

        # List and sort all model paths
        model_filenames = os.listdir(models_dirpath)
        model_paths = sorted(os.path.join(models_dirpath, fname) for fname in model_filenames)
        logging.info("Loading %d models...", len(model_paths))

        # Set random seed for reproducibility
        random_seed = 44
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Load models and ground truth
        model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_paths)

        # Extract features and labels
        X = np.vstack([
            self.generate_features(model_dict).numpy()
            for model_dict in model_repr_dict['Box']
        ])
        y = np.array(model_ground_truth_dict['Box'])

        # print("Number of NaNs in X:", np.isnan(X).sum())
        # print("Number of Infs in X:", np.isinf(X).sum())
        # exit(0)


        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=random_seed
        )
        self.input_features = X_train.shape[1]

        # Convert to PyTorch tensors
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).long()
        X_val_tensor = torch.from_numpy(X_val).float()
        y_val_tensor = torch.from_numpy(y_val).long()

        # Reshape tensors to match the expected input shape of the model
        X_train_tensor = X_train_tensor.unsqueeze(1)  # Add a channel dimension
        X_val_tensor = X_val_tensor.unsqueeze(1)

        # Create TensorDatasets and DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Define the Inception model components
        import torch.nn.functional as F

        def correct_sizes(sizes):
            corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
            return corrected_sizes

        def pass_through(X):
            return X

        class Inception(nn.Module):
            def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU(), return_indices=False):
                super(Inception, self).__init__()
                self.return_indices = return_indices
                if in_channels > 1:
                    self.bottleneck = nn.Conv1d(
                        in_channels=in_channels, 
                        out_channels=bottleneck_channels, 
                        kernel_size=1, 
                        stride=1, 
                        bias=False
                    )
                else:
                    self.bottleneck = nn.Identity()
                    bottleneck_channels = in_channels

                self.conv_from_bottleneck_1 = nn.Conv1d(
                    in_channels=bottleneck_channels, 
                    out_channels=n_filters, 
                    kernel_size=kernel_sizes[0], 
                    stride=1, 
                    padding=kernel_sizes[0]//2, 
                    bias=False
                )
                self.conv_from_bottleneck_2 = nn.Conv1d(
                    in_channels=bottleneck_channels, 
                    out_channels=n_filters, 
                    kernel_size=kernel_sizes[1], 
                    stride=1, 
                    padding=kernel_sizes[1]//2, 
                    bias=False
                )
                self.conv_from_bottleneck_3 = nn.Conv1d(
                    in_channels=bottleneck_channels, 
                    out_channels=n_filters, 
                    kernel_size=kernel_sizes[2], 
                    stride=1, 
                    padding=kernel_sizes[2]//2, 
                    bias=False
                )
                self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
                self.conv_from_maxpool = nn.Conv1d(
                    in_channels=in_channels, 
                    out_channels=n_filters, 
                    kernel_size=1, 
                    stride=1,
                    padding=0, 
                    bias=False
                )
                self.batch_norm = nn.BatchNorm1d(num_features=4*n_filters)
                self.activation = activation

            def forward(self, X):
                Z_bottleneck = self.bottleneck(X)
                if self.return_indices:
                    Z_maxpool, indices = self.max_pool(X)
                else:
                    Z_maxpool = self.max_pool(X)
                Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
                Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
                Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
                Z4 = self.conv_from_maxpool(Z_maxpool)
                Z = torch.cat([Z1, Z2, Z3, Z4], dim=1)
                Z = self.activation(self.batch_norm(Z))
                if self.return_indices:
                    return Z, indices
                else:
                    return Z

        class InceptionBlock(nn.Module):
            def __init__(self, in_channels, n_filters=32, kernel_sizes=[9,19,39], bottleneck_channels=32, use_residual=True, activation=nn.ReLU(), return_indices=False):
                super(InceptionBlock, self).__init__()
                self.use_residual = use_residual
                self.return_indices = return_indices
                self.activation = activation
                self.inception_1 = Inception(
                    in_channels=in_channels,
                    n_filters=n_filters,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    activation=activation,
                    return_indices=return_indices
                )
                self.inception_2 = Inception(
                    in_channels=4*n_filters,
                    n_filters=n_filters,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    activation=activation,
                    return_indices=return_indices
                )
                self.inception_3 = Inception(
                    in_channels=4*n_filters,
                    n_filters=n_filters,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    activation=activation,
                    return_indices=return_indices
                )
                if self.use_residual:
                    self.residual = nn.Sequential(
                        nn.Conv1d(
                            in_channels=in_channels, 
                            out_channels=4*n_filters, 
                            kernel_size=1,
                            stride=1,
                            padding=0
                        ),
                        nn.BatchNorm1d(
                            num_features=4*n_filters
                        )
                    )

            def forward(self, X):
                residual = X
                if self.return_indices:
                    Z, i1 = self.inception_1(X)
                    Z, i2 = self.inception_2(Z)
                    Z, i3 = self.inception_3(Z)
                else:
                    Z = self.inception_1(X)
                    Z = self.inception_2(Z)
                    Z = self.inception_3(Z)
                if self.use_residual:
                    if residual.shape != Z.shape:
                        residual = self.residual(residual)
                    Z = Z + residual
                    Z = self.activation(Z)
                if self.return_indices:
                    return Z, [i1, i2, i3]
                else:
                    return Z

        class Flatten(nn.Module):
            def forward(self, x):
                return x.view(x.size(0), -1)

        # Define the model
        input_channels = 1  # Since we have added a channel dimension
        filters = 32  # You can adjust this as needed

        net = nn.Sequential(
            InceptionBlock(
                in_channels=input_channels,
                n_filters=filters,
                kernel_sizes=correct_sizes([5, 11, 23]),
                bottleneck_channels=filters,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=filters*4, 
                n_filters=filters,
                kernel_sizes=correct_sizes([5, 11, 23]),
                bottleneck_channels=filters,
                use_residual=True,
                activation=nn.ReLU()
            ),
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(),
            nn.Linear(in_features=4*filters*1, out_features=len(np.unique(y)))  # Adjust output features based on number of classes
        )

        # Move the model to the appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        # optimizer = optim.Adam(net.parameters(), lr=0.002)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300], gamma=0.1)

        # Training loop with validation
        num_epochs = 400

        for epoch in range(num_epochs):
            net.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            scheduler.step()

            epoch_loss = running_loss / len(train_dataset)

            # Validation phase
            net.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_accuracy = correct / total * 100
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Save the trained model
        logging.info("Saving model...")
        torch.save(net.state_dict(), self.model_filepath)

        self.write_metaparameters()
        logging.info("Configuration done!")


    def inference_on_example_data(self, model_filepath, examples_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model_filepath: path to the pytorch model file
            examples_dirpath: the directory path for the round example data
        """

        model_dirpath = os.path.dirname(model_filepath)
        full_config = os.path.join(model_dirpath, "reduced-config.json")
        print(full_config)
        sys.stdout.flush()
        with open(full_config, 'r') as f1:
            config = Box(json.load(f1))

        config.seed = 0
        config.log_folder = "./logs/temp"
        config.model_folder = "./output/temp"
        config.render_mode = "rgb_array"

        envs = [make_clean_env]

        opac_object = OffPolicyActorCriticMultitask(
            config,
            make_train_envs=envs,
            make_test_envs=envs,
        )

        opac_object.initialize_env()
        opac_object.initialize_networks()
        opac_object.pi_network.load_state_dict(torch.load(model_filepath)["pi"])
        opac_object.sampler = get_sampler(opac_object.config.pi_network)

        # ========================================================================

        display_render = False #True
        delay = 0.01
        num_episodes = 5
        env = opac_object.env
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for j in range(num_episodes):
            obs, info = opac_object.reset_env(env)
            terminated, truncated = False, False
            returns = 0.0
            success = False
            while not terminated and not truncated:
                action = opac_object.get_action(obs, deterministic=True)
                with torch.no_grad():
                    torch_obs = torch.from_numpy(obs).to(device).float()
                    torch_act = torch.from_numpy(action).to(device).float()
                    pi = opac_object.pi_network(torch_obs)
                    _, log_prob = opac_object.sampler.get_action_and_log_prob(pi)
                next_obs, reward, terminated, truncated, info = env.step(action)
                returns += reward

                if display_render:
                    rgb = env.render()
                    cv2.imshow("render", rgb[:, :, ::-1])
                    cv2.waitKey(1)

                if delay > 0.0:
                    time.sleep(delay)

                obs = next_obs

                if "success" in info and info["success"]:
                    success = True

            print("Success:", success, "Total reward:", returns)

    def infer(
            self,
            model_filepath,
            result_filepath,
            scratch_dirpath,
            examples_dirpath,
            round_training_dataset_dirpath,
    ):
        """Method to predict whether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        # Inferences on examples to demonstrate how it is done for a round
        self.inference_on_example_data(model_filepath, examples_dirpath)

        # Build a feature vector for the model to compute its probability of poisoning
        _, model_repr_dict, _ = load_model(model_filepath)
        X = self.generate_features(model_repr_dict)

        # Load the InceptionTime model from the learned-parameters location
        import torch
        import torch.nn as nn

        # Define the InceptionTime model architecture (must match the trained model)
        def correct_sizes(sizes):
            corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
            return corrected_sizes

        class Inception(nn.Module):
            def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU(), return_indices=False):
                super(Inception, self).__init__()
                self.return_indices = return_indices
                if in_channels > 1:
                    self.bottleneck = nn.Conv1d(
                        in_channels=in_channels, 
                        out_channels=bottleneck_channels, 
                        kernel_size=1, 
                        stride=1, 
                        bias=False
                    )
                else:
                    self.bottleneck = nn.Identity()
                    bottleneck_channels = in_channels

                self.conv_from_bottleneck_1 = nn.Conv1d(
                    in_channels=bottleneck_channels, 
                    out_channels=n_filters, 
                    kernel_size=kernel_sizes[0], 
                    stride=1, 
                    padding=kernel_sizes[0] // 2, 
                    bias=False
                )
                self.conv_from_bottleneck_2 = nn.Conv1d(
                    in_channels=bottleneck_channels, 
                    out_channels=n_filters, 
                    kernel_size=kernel_sizes[1], 
                    stride=1, 
                    padding=kernel_sizes[1] // 2, 
                    bias=False
                )
                self.conv_from_bottleneck_3 = nn.Conv1d(
                    in_channels=bottleneck_channels, 
                    out_channels=n_filters, 
                    kernel_size=kernel_sizes[2], 
                    stride=1, 
                    padding=kernel_sizes[2] // 2, 
                    bias=False
                )
                self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
                self.conv_from_maxpool = nn.Conv1d(
                    in_channels=in_channels, 
                    out_channels=n_filters, 
                    kernel_size=1, 
                    stride=1,
                    padding=0, 
                    bias=False
                )
                self.batch_norm = nn.BatchNorm1d(num_features=4 * n_filters)
                self.activation = activation

            def forward(self, X):
                Z_bottleneck = self.bottleneck(X)
                if self.return_indices:
                    Z_maxpool, indices = self.max_pool(X)
                else:
                    Z_maxpool = self.max_pool(X)
                Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
                Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
                Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
                Z4 = self.conv_from_maxpool(Z_maxpool)
                Z = torch.cat([Z1, Z2, Z3, Z4], dim=1)
                Z = self.activation(self.batch_norm(Z))
                if self.return_indices:
                    return Z, indices
                else:
                    return Z

        class InceptionBlock(nn.Module):
            def __init__(self, in_channels, n_filters=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32, use_residual=True, activation=nn.ReLU(), return_indices=False):
                super(InceptionBlock, self).__init__()
                self.use_residual = use_residual
                self.return_indices = return_indices
                self.activation = activation
                self.inception_1 = Inception(
                    in_channels=in_channels,
                    n_filters=n_filters,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    activation=activation,
                    return_indices=return_indices
                )
                self.inception_2 = Inception(
                    in_channels=4 * n_filters,
                    n_filters=n_filters,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    activation=activation,
                    return_indices=return_indices
                )
                self.inception_3 = Inception(
                    in_channels=4 * n_filters,
                    n_filters=n_filters,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    activation=activation,
                    return_indices=return_indices
                )
                if self.use_residual:
                    self.residual = nn.Sequential(
                        nn.Conv1d(
                            in_channels=in_channels, 
                            out_channels=4 * n_filters, 
                            kernel_size=1,
                            stride=1,
                            padding=0
                        ),
                        nn.BatchNorm1d(
                            num_features=4 * n_filters
                        )
                    )

            def forward(self, X):
                residual = X
                if self.return_indices:
                    Z, i1 = self.inception_1(X)
                    Z, i2 = self.inception_2(Z)
                    Z, i3 = self.inception_3(Z)
                else:
                    Z = self.inception_1(X)
                    Z = self.inception_2(Z)
                    Z = self.inception_3(Z)
                if self.use_residual:
                    if residual.shape != Z.shape:
                        residual = self.residual(residual)
                    Z = Z + residual
                    Z = self.activation(Z)
                if self.return_indices:
                    return Z, [i1, i2, i3]
                else:
                    return Z

        class Flatten(nn.Module):
            def forward(self, x):
                return x.view(x.size(0), -1)

        # Define the model (ensure parameters match those used during training)
        input_channels = 1  # Since we added a channel dimension during training
        filters = 32  # Must match the number used during training
        num_classes = 2  # Adjust based on your problem

        net = nn.Sequential(
            InceptionBlock(
                in_channels=input_channels,
                n_filters=filters,
                kernel_sizes=correct_sizes([5, 11, 23]),
                bottleneck_channels=filters,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=filters * 4, 
                n_filters=filters,
                kernel_sizes=correct_sizes([5, 11, 23]),
                bottleneck_channels=filters,
                use_residual=True,
                activation=nn.ReLU()
            ),
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(),
            nn.Linear(in_features=4 * filters * 1, out_features=num_classes)
        )

        # Load the saved model parameters
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.to(device)
        net.load_state_dict(torch.load(self.model_filepath, map_location=device))
        net.eval()  # Set the model to evaluation mode

        # Prepare the input data
        import numpy as np

        X_tensor = X.float()
        X_tensor = X_tensor.unsqueeze(1)  # Adds channel dimension
        X_tensor = X_tensor.to(device)

        # Run inference
        with torch.no_grad():
            outputs = net(X_tensor)
            probabilities = nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            poisoning_probability = probabilities[0, 1].item()  # Assuming class '1' corresponds to 'poisoned'

        # use the RandomForest to predict the trojan probability based on the feature vector X
        probability = poisoning_probability # regressor.predict(X)[0]
        # clip the probability to reasonable values
        probability = np.clip(probability, a_min=0.01, a_max=0.99)

        # write the trojan probability to the output file
        with open(result_filepath, "w") as fp:
            fp.write(str(probability))

        logging.info("Trojan probability: {}".format(probability))
