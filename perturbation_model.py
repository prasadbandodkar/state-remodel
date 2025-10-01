# Base class to build a perturbation model

# Basic imports
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, List, Dict

# PyTorch imports
import torch
import torch.nn as nn
import lightning as L


@dataclass
class PerturbationModel(ABC, L.LightningModule):
    input_dim: int
    output_dim: int
    hidden_dim: int
    
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
        super().__init__()
    
    @abstractmethod
    def _build_networks(self):
        """ Build the core neural network components """
        pass
    
    def _build_decoder(self):
        """ Build self.gene_decoder from self.decoder_config"""
        pass
    
    
    # Configure steps
    def training_step(self, batch: Dict[str, torch.Tensor], batch_index: int) -> torch.Tensor:
        """ Training step logic for both main model and decoder"""
        pass
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_index: int) -> torch.Tensor:
        """ Validation step logic for both main model and decoder """
        pass
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_index: int) -> torch.Tensor:
        """ Test step logic for both main model and decoder """
        pass
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_index: int) -> torch.Tensor:
        """ Predict step logic for both main model and decoder """
        pass
