# Class to build the transformer backbone

# Basic imports
from dataclasses import dataclass
from typing import Optional, List, Dict
from abc import ABC, abstractmethod

# Deep learning imports
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
from transformers import LlamaConfig, LlamaModel
from transformers import PreTrainedModel


@dataclass
class Llama(LlamaModel):
    """Wrapper around LlamaModel that:
    - Disables the causal mask
    - Allows full bidirectional attention
    - Prints internal bias mask each forward pass
    """
    config: LlamaConfig
    
    def __post_init__(self):
        super().__init__(self.config)
        
        self.rotary_emb = NoRoPE(
            head_dim=self.config.head_dim,
        )
        
        self.config.is_causal = False
        
        for layer in self.layers:
            if (hasattr(layer, "self_attn")):
                layer.self_attn.is_causal = False
    
    def _update_causal_mask()
        
        

    

@dataclass
class GPT2(GPT2Model):
    """Wrapper around GPT2Model that:
    - Disables the causal mask
    - Allows full bidirectional attention
    - Prints internal bias mask each forward pass

    Args:
        GPT2Model (_type_): _description_
    """
    config: GPT2Config
    base: GPT2Model
    
    def __post_init__(self):
        super().__init__(self.config)
        
        for block in self.h:
            block.attn.bias.data.fill_(True)
            block.attn.is_causal = False
        
        
        
