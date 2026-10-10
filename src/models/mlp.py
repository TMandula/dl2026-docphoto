"""A multilayer perceptron on raw pixels.

TASK 2. Build `MLP`: a fully connected network that maps one flattened,
standardised photograph (`in_dim` grey levels) to one number in [0, 1].

* `hidden` is a list of layer widths from the config, for example [256, 64];
  build one Linear + nonlinearity + Dropout block per entry.
* `dropout` is the probability from the config. Where it goes, and whether
  it belongs after the last hidden layer, is your call.
* The output must lie in [0, 1]. `prior` is the mean target of the training
  rows; it is passed to you for a reason you have to find yourself.
* `forward` returns shape (batch,), not (batch, 1).

It trains in seconds on a CPU. That is the point of starting here: every
decision in it can be tested in the time it takes to read this docstring.
Count its parameters against the number of training photographs before you
decide how much dropout it needs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float = 0.3, prior: float = 0.5):
        super().__init__()
        # 1. Building hidden layers (Linear + nonlinearity + Dropout)
        layers = []
        current_dim = in_dim
        for h_dim in hidden:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = h_dim
            
        self.hidden_layers = nn.Sequential(*layers)
        
        # 2. Final output layer (for 1 dimension)
        self.output_layer = nn.Linear(current_dim, 1)
        
        # 3. Setting the output bias to the prior logit value,
        # so that the untrained network yields the correct average approximation right from the start.
        with torch.no_grad():
            # Safety guard to avoid division by zero or one
            p = max(min(prior, 0.999), 0.001)
            bias_val = torch.log(torch.tensor(p / (1 - p)))
            self.output_layer.bias.fill_(bias_val)
            
        # 4. Sigmoid activation constraining the output to the [0, 1] range
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Execution on the hidden layers
        out = self.hidden_layers(x)
        # Output layer
        out = self.output_layer(out)
        # Mapping value to [0, 1]
        out = self.sigmoid(out)
        # Adjusting the output size from (batch, 1) to (batch,)
        return out.squeeze(-1)
