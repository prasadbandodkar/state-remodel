# Class to build the STATE transition model

# Basic imports
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# PyTorch imports
import torch
import torch.nn as nn

# Custom imports
from perturbation_model import PerturbationModel
from mlp import MLP


@dataclass
class StateTransitionPerturbationModel(PerturbationModel):
    """

    Args:
        PerturbationModel (_type_): _description_
    """
    input_dim: int     # dimension of the input expression - #genes or embedding dimension
    hidden_dim: int
    output_dim: int    # dimension of the output space - genes or latent
    
    
    # Perturbation and batch dimensions
    pert_dim: int
    batch_dim: int
    gene_dim: int = 5000
    hvg_dim : int = 64
    
    # Model params
    dropout_prob: float = 0.1
    batch_size: int = 64
    
    # Training params
    lr: float = 3e-4
    loss_fn: nn.Module = nn.MSELoss()
    
    # Biological params
    output_space: str = "gene"
    embed_key: str = None
    gene_names = Optional[List[str]] = None
    
    # Decoder params
    decoder_config: dict | None = None
    
    def __post_init__(self, **kwargs):
        super().__init__(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            gene_dim=self.gene_dim,
            output_dim=self.output_dim,
            pert_dim=self.pert_dim,
            batch_dim=self.batch_dim,
            output_space=self.output_space,
            **kwargs,
        )
        
        # Transfomer specific params
        self.n_encoder_layers = kwargs.get("n_encoder_layers", 2)
        self.n_decoder_layers = kwargs.get("n_decoder_layers", 2)

        # Build the newural network
        self._build_networks()
        
    
    def _build_networks(self) -> None:
        """ Build the core neural network components """
        # Perturbation encoder
        self.pert_encoder = MLP(
            input_dim=self.pert_dim,
            output_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.n_encoder_layers,
            dropout_prob=self.dropout_prob,
            activation=self.activation_class,
        )
        
        # Cell encoder
        self.cell_encoder = MLP(
            input_dim=self.input_dim,
            output_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.n_encoder_layers,
            dropout_prob=self.dropout_prob,
            activation=self.activation_class,
        )
        
        # Tranformer backbone
    
    
    def forward():
        pass
    
    def training_step():
        pass
    
    def validation_step():
        pass
    
    def test_step():
        pass
    
    def predict_step():
        pass
        
        
        
            