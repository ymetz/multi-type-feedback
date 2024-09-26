"""Module for instantiating a neural network."""

# pylint: disable=arguments-differ
from typing import Callable, Type, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import log_softmax, nll_loss
from pytorch_lightning import LightningModule
from masksembles.torch import Masksembles1D, Masksembles2D
from torch import Tensor, nn
from torch.nn.functional import mse_loss
import gymnasium as gym

# Loss functions
single_reward_loss = nn.MSELoss()

def calculate_mse_loss(network: LightningModule, batch: Tensor):
    """Calculate the mean squared erro loss for the reward."""
    return mse_loss(network(batch[0]), batch[1].unsqueeze(1), reduction="sum")


def calculate_mle_loss(network: LightningModule, batch: Tensor):
    """Calculate the maximum likelihood loss for the better trajectory."""
    rewards1 = network(batch[0]).flatten()
    rewards2 = network(batch[1]).flatten()

    probs_softmax = torch.exp(rewards1) / (torch.exp(rewards1) + torch.exp(rewards2))

    loss = -torch.sum(torch.log(probs_softmax))

    return loss


def calculate_pairwise_loss(network: LightningModule, batch: Tensor):
    """Calculate the maximum likelihood loss for the better trajectory."""
    (pair_obs, pair_actions), preferred_indices = batch  # preferred_indices: (batch_size,)

    # Unpack observations and actions for both trajectories
    obs1, obs2 = pair_obs[0], pair_actions[0]
    actions1, actions2 = pair_obs[1], pair_actions[1]

    # Compute network outputs
    outputs1 = network(obs1, actions1)  # Shape: (batch_size, segment_length, output_dim)
    outputs2 = network(obs2, actions2)

    # Sum over sequence dimension
    rewards1 = outputs1.sum(dim=1).squeeze(-1)  # Shape: (batch_size,)
    rewards2 = outputs2.sum(dim=1).squeeze(-1)

    # Stack rewards and compute log softmax
    rewards = torch.stack([rewards1, rewards2], dim=1)  # Shape: (batch_size, 2)
    log_probs = F.log_softmax(rewards, dim=1)

    # Compute NLL loss
    loss = nll_loss(log_probs, preferred_indices)

    return loss


def calculate_single_reward_loss(network: LightningModule, batch: Tensor):
    """Calculate the MSE loss between prediction and actual reward."""
    (observations, actions), targets = batch
    # Network output: (batch_size, segment_length, output_dim)
    outputs = network(observations, actions)

    # Sum over the sequence dimension to get total rewards per segment
    total_rewards = outputs.sum(dim=1)  # Shape: (batch_size, output_dim)

    # Ensure targets have the correct shape
    targets = targets.float().unsqueeze(1)  # Shape: (batch_size, 1)

    # Compute loss
    loss = single_reward_loss(total_rewards, targets)

    return loss


# Lightning networks


class LightningNetwork(LightningModule):
    """Neural network to model the RL agent's reward using Pytorch Lightning."""

    def __init__(
        self,
        input_spaces: Tuple[gym.spaces.Space, gym.spaces.Space],
        layer_num: int,
        output_dim: int,
        hidden_dim: int,
        action_hidden_dim: int, # not used here
        loss_function: Callable[[LightningModule, Tensor], Tensor],
        learning_rate: float,
        activation_function: Type[nn.Module] = nn.ReLU,
        last_activation: Union[Type[nn.Module], None] = None,
        ensemble_count: int = 0,
    ):
        super().__init__()

        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.ensemble_count = ensemble_count
        obs_space, action_space = input_spaces
        input_dim = obs_space.shape[-1] + action_space.shape[-1]

        # Initialize the network
        layers_unit = [input_dim] + [hidden_dim] * (layer_num - 1)

        layers: list[nn.Module] = []

        for idx in range(len(layers_unit) - 1):
            layers.append(nn.Linear(layers_unit[idx], layers_unit[idx + 1]))
            layers.append(activation_function())

            if self.ensemble_count > 1:
                layers.append(
                    Masksembles1D(
                        channels=layers_unit[idx + 1],
                        n=self.ensemble_count,
                        scale=1.8,
                    ).float()
                )

        layers.append(nn.Linear(layers_unit[-1], output_dim))

        if last_activation is not None:
            layers.append(last_activation())

            if self.ensemble_count > 1:
                layers.append(
                    Masksembles1D(
                        channels=output_dim, n=self.ensemble_count, scale=1.8
                    ).float()
                )

        self.network = nn.Sequential(*layers)

        # Initialize the weights
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)

                layer.bias.data.zero_()

        self.save_hyperparameters()

    def forward(self, observations: Tensor, actions: Tensor):
        """Do a forward pass through the neural network (inference)."""
        # observations: (batch_size, segment_length, obs_dim)
        # actions: (batch_size, segment_length, action_dim)
        batch_size, segment_length, obs_dim = observations.shape
        _, _, action_dim = actions.shape

        # Flatten the batch and sequence dimensions
        obs_flat = observations.reshape(-1, obs_dim)
        actions_flat = actions.reshape(-1, action_dim)

        # Concatenate observations and actions
        batch = torch.cat((obs_flat, actions_flat), dim=1)  # Shape: (batch_size * segment_length, obs_dim + action_dim)

        # Pass through the network
        output = self.network(batch)  # Shape: (batch_size * segment_length, output_dim)

        # Reshape back to (batch_size, segment_length, output_dim)
        output = output.reshape(batch_size, segment_length, -1)

        return output

    def training_step(self, batch: Tensor):
        """Compute the loss for training."""
        loss = self.loss_function(self, batch)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: Tensor, _batch_idx: int):
        """Compute the loss for validation."""
        loss = self.loss_function(self, batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer to optimize the neural network."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


class LightningCnnNetwork(LightningModule):
    """Neural network to model the RL agent's reward using Pytorch Lightning,
    based on the Impapala CNN architecture. Use given layer_num, with a single
    fully connected layer"""

    def __init__(
        self,
        input_spaces: Tuple[gym.spaces.Space, gym.spaces.Space],
        layer_num: int,
        output_dim: int,
        hidden_dim: int, # not used here
        action_hidden_dim: int,
        loss_function: Callable[[LightningModule, Tensor], Tensor],
        learning_rate: float,
        cnn_channels: list[int] = (16, 32, 32),
        activation_function: Type[nn.Module] = nn.ReLU,
        last_activation: Union[Type[nn.Module], None] = None,
        ensemble_count: int = 0,
    ):
        super().__init__()

        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.ensemble_count = ensemble_count
        obs_space, action_space = input_spaces
        input_channels = obs_space.shape[0]

        # Initialize the network
        layers = []
        for i in range(layer_num):
            layers.append(self.conv_sequence(input_channels, cnn_channels[i]))
            input_channels = cnn_channels[i]

        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        
        # action input_layer
        action_shape = action_space.shape if action_space.shape else 1
        self.action_in = nn.Linear(action_shape, action_hidden_dim)
        
        self.fc = nn.Linear(
            self.compute_flattened_size(obs_space.shape, cnn_channels) + action_hidden_dim, output_dim
        )
        self.masksemble_out = Masksembles1D(4, 1.8)

        self.save_hyperparameters()

    def conv_layer(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def residual_block(self, in_channels):
        return nn.Sequential(
            nn.ReLU(),
            Masksembles2D(in_channels, 4, 1.8),
            self.conv_layer(in_channels, in_channels),
            nn.ReLU(),
            Masksembles2D(in_channels, 4, 1.8),
            self.conv_layer(in_channels, in_channels),
        )

    def conv_sequence(self, in_channels, out_channels):
        return nn.Sequential(
            self.conv_layer(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.residual_block(out_channels),
            self.residual_block(out_channels),
        )

    def compute_flattened_size(self, observation_space, cnn_channels):
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space).squeeze(-1)
            sample_output = self.conv_layers(sample_input)
            return int(np.prod(sample_output.size()))

    def forward(self, observations, actions):
        # observations: (batch_size, segment_length, channels, height, width)
        # actions: (batch_size, segment_length, action_dim)
        batch_size, segment_length, channels, height, width = observations.shape
        _, _, action_dim = actions.shape

        # Flatten the batch and sequence dimensions for observations
        obs_flat = observations.reshape(-1, channels, height, width)

        # Process observations through convolutional layers
        x = self.conv_layers(obs_flat)
        x = self.flatten(x)
        x = F.relu(x)

        # Flatten actions
        actions_flat = actions.reshape(-1, action_dim)
        act = self.action_in(actions_flat)
        act = F.relu(act)

        # Concatenate processed observations and actions
        x = torch.cat((x, act), dim=1)
        x = self.fc(x)  # Shape: (batch_size * segment_length, output_dim)
        x = self.masksemble_out(x)

        # Reshape back to (batch_size, segment_length, output_dim)
        x = x.reshape(batch_size, segment_length, -1)

        return x

    def training_step(self, batch: Tensor):
        """Compute the loss for training."""
        loss = self.loss_function(self, batch)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: Tensor, _batch_idx: int):
        """Compute the loss for validation."""
        loss = self.loss_function(self, batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer to optimize the neural network."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
