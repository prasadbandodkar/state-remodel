# Class to build an MLP

# Basic imports
from dataclasses import dataclass

# PyTorch imports
import torch
import torch.nn as nn


@dataclass
class MLP(nn.Module):
    input_dim: int
    output_dim: int
    hidden_dim: int
    num_layers: int
    dropout_prob: float = 0.0
    activation: nn.Module = nn.ReLU # nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU, nn.GELU
    
    def __post_init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        if self.num_layers == 1:
            # Single layer: input -> output directly
            self.layers.append(nn.Linear(self.input_dim, self.output_dim))
        else:
            # Multiple layers: input -> hidden -> ... -> hidden -> output
            self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            for _ in range(self.num_layers - 1):
                self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.activation()(x)
                x = nn.Dropout(self.dropout_prob)(x)
        return x
    
    