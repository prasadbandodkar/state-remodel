# Mathematical structure of state transition models
_Exported on 9/24/2025 at 10:39:53 PDT from Cursor (1.6.45)_

---

**User**

Explain mathematically how the state transition model is structured.

---

**Cursor**

I'll analyze the state transition model structure mathematically by examining the code. Let me first understand the overall architecture and then break down the mathematical components.


Now let me examine the utils file to understand the transformer backbone and other components:


Let me get the transformer backbone function:


Now let me look at the decoders to understand the complete architecture:


Now I have enough information to provide a comprehensive mathematical explanation of the state transition model. Let me structure this analysis:

## Mathematical Structure of the State Transition Perturbation Model

Based on the code analysis, the `StateTransitionPerturbationModel` is a sophisticated neural architecture that models cellular state transitions under perturbations. Here's the mathematical breakdown:

### 1. **Overall Architecture**

The model implements a **set-to-set function** that maps from control cell states to perturbed cell states:

```
f: (X_control, P) → X_perturbed
```

Where:
- `X_control ∈ ℝ^(B×S×D_input)` are control cell embeddings
- `P ∈ ℝ^(B×S×D_pert)` are perturbation embeddings  
- `X_perturbed ∈ ℝ^(B×S×D_output)` are predicted perturbed cell embeddings
- `B` = batch size, `S` = sequence length (cell_sentence_len), `D_*` = dimensions

### 2. **Input Encoding**

The model first encodes both control cells and perturbations into a shared latent space:

**Perturbation Encoding:**
```
E_pert = MLP_pert(P) ∈ ℝ^(B×S×D_hidden)
```

**Control Cell Encoding:**
```
E_control = MLP_basal(X_control) ∈ ℝ^(B×S×D_hidden)
```

Where `MLP_pert` and `MLP_basal` are multi-layer perceptrons with:
- `n_encoder_layers` hidden layers
- Hidden dimension `D_hidden`
- Activation function (GELU by default)
- Dropout for regularization

### 3. **Combined Input and Optional Batch Encoding**

The encoded inputs are combined additively:
```
E_combined = E_pert + E_control ∈ ℝ^(B×S×D_hidden)
```

If batch information is available:
```
E_final = E_combined + Embedding_batch(batch_indices) ∈ ℝ^(B×S×D_hidden)
```

### 4. **Confidence Token (Optional)**

If enabled, a learnable confidence token is appended:
```
E_with_confidence = [E_final, c_token] ∈ ℝ^(B×(S+1)×D_hidden)
```

Where `c_token ∈ ℝ^(1×1×D_hidden)` is a learnable parameter.

### 5. **Transformer Backbone**

The core of the model uses a transformer (GPT-2 or LLaMA) to process the sequence:

```
H = Transformer(E_with_confidence) ∈ ℝ^(B×(S+1)×D_hidden)
```

The transformer enables **cross-cell attention**, allowing cells to attend to one another and learn set-to-set relationships rather than independent cell mappings.

### 6. **Output Projection**

The transformer output is projected to the desired output dimension:

```
H_main = H[:, :-1, :] ∈ ℝ^(B×S×D_hidden)  # Main output
H_confidence = H[:, -1, :] ∈ ℝ^(B×1×D_hidden)  # Confidence output (if used)
```

**Main prediction:**
```
Y = MLP_out(H_main) ∈ ℝ^(B×S×D_output)
```

### 7. **Residual Prediction (Optional)**

If `predict_residual = True`:
```
Y_final = Y + X_control  # Additive residual
```

### 8. **Output Space Processing**

Depending on the output space:
- **Gene space**: Apply ReLU activation for non-negative gene expression
- **Latent space**: No additional processing

### 9. **Loss Functions**

The model supports multiple loss functions:

**Energy Loss (default):**
```
L_energy = EnergyLoss(Y_pred, Y_target)
```

**Sinkhorn Loss:**
```
L_sinkhorn = SinkhornLoss(Y_pred, Y_target)
```

**Combined Loss:**
```
L_combined = α·L_sinkhorn + β·L_energy
```

Where `α` and `β` are weighting parameters.

### 10. **Confidence Prediction (Optional)**

If confidence token is enabled:
```
c_pred = MLP_confidence(H_confidence) ∈ ℝ^(B×1)
L_confidence = MSE(c_pred, L_main.detach())
```

### 11. **Gene Decoder (Optional)**

For mapping back to gene expression space:
```
X_genes = GeneDecoder(Y_final) ∈ ℝ^(B×S×D_genes)
```

Where `GeneDecoder` is an MLP that maps from latent space to gene expression space.

### 12. **Regularization (Optional)**

L1 regularization on the perturbation effect:
```
L_reg = λ·||Y_pred - X_control||_1
```

### **Key Mathematical Properties**

1. **Set-to-Set Mapping**: The transformer architecture enables cells to attend to each other, learning global patterns rather than independent mappings.

2. **Distributional Loss**: Uses optimal transport-based losses (energy, Sinkhorn) that are well-suited for comparing distributions of cells.

3. **Residual Learning**: Can predict either absolute states or residual changes from control.

4. **Multi-Scale Processing**: Handles both latent space and gene expression space through optional decoders.

5. **Uncertainty Quantification**: Optional confidence token provides uncertainty estimates.

This architecture is particularly well-suited for modeling cellular state transitions because it can capture both local cell-specific effects and global population-level patterns through the attention mechanism.

```plaintext
f: (X_control, P) → X_perturbed
```

```plaintext
E_pert = MLP_pert(P) ∈ ℝ^(B×S×D_hidden)
```

```plaintext
E_control = MLP_basal(X_control) ∈ ℝ^(B×S×D_hidden)
```

```plaintext
E_combined = E_pert + E_control ∈ ℝ^(B×S×D_hidden)
```

```plaintext
E_final = E_combined + Embedding_batch(batch_indices) ∈ ℝ^(B×S×D_hidden)
```

```plaintext
E_with_confidence = [E_final, c_token] ∈ ℝ^(B×(S+1)×D_hidden)
```

```plaintext
H = Transformer(E_with_confidence) ∈ ℝ^(B×(S+1)×D_hidden)
```

```plaintext
H_main = H[:, :-1, :] ∈ ℝ^(B×S×D_hidden)  # Main output
H_confidence = H[:, -1, :] ∈ ℝ^(B×1×D_hidden)  # Confidence output (if used)
```

```plaintext
Y = MLP_out(H_main) ∈ ℝ^(B×S×D_output)
```

```plaintext
Y_final = Y + X_control  # Additive residual
```

```plaintext
L_energy = EnergyLoss(Y_pred, Y_target)
```

```plaintext
L_sinkhorn = SinkhornLoss(Y_pred, Y_target)
```

```plaintext
L_combined = α·L_sinkhorn + β·L_energy
```

```plaintext
c_pred = MLP_confidence(H_confidence) ∈ ℝ^(B×1)
L_confidence = MSE(c_pred, L_main.detach())
```

```plaintext
X_genes = GeneDecoder(Y_final) ∈ ℝ^(B×S×D_genes)
```

```plaintext
L_reg = λ·||Y_pred - X_control||_1
```

---

**User**

Explain how the control flows - which class to which class.

---

**Cursor**

I'll analyze the control flow by examining the class hierarchy and how data flows through the different components of the state transition model.


Now let me examine the main entry point and how the models are instantiated:


Now let me examine the CLI functions to understand how models are instantiated:


Let me continue reading the training function to see how models are instantiated:


Now let me check the `get_lightning_module` function to see how models are instantiated:


Now let me examine some of the other model classes to understand the inheritance hierarchy:


Now let me create a comprehensive control flow diagram. Based on my analysis, here's how the control flows through the different classes:

## Control Flow Analysis: State Transition Model Architecture

### **1. Entry Point and Configuration Flow**

```
state/__main__.py
├── get_args() → parse command line arguments
├── load_hydra_config() → load configuration from YAML files
└── main() → route to appropriate command handler
    └── case "tx" → run_tx_train(cfg)
```

### **2. Training Pipeline Flow**

```
state/_cli/_tx/_train.py
├── run_tx_train(cfg)
├── get_datamodule() → create data module
├── get_lightning_module() → instantiate model
├── get_loggers() → setup logging
├── get_checkpoint_callbacks() → setup checkpointing
└── trainer.fit() → start training
```

### **3. Model Instantiation Flow**

```
state/tx/utils/__init__.py
└── get_lightning_module(model_type, data_config, model_config, training_config, var_dims)
    ├── case "neuralot" | "pertsets" | "state" → StateTransitionPerturbationModel
    ├── case "embedsum" → EmbedSumPerturbationModel  
    ├── case "old_neuralot" → OldNeuralOTPerturbationModel
    ├── case "perturb_mean" → PerturbMeanPerturbationModel
    ├── case "context_mean" → ContextMeanPerturbationModel
    ├── case "decoder_only" → DecoderOnlyPerturbationModel
    ├── case "pseudobulk" → PseudobulkPerturbationModel
    ├── case "cpa" → CPAPerturbationModel
    ├── case "scvi" → SCVIPerturbationModel
    └── case "scgpt-*" → scGPTForPerturbation
```

### **4. Class Hierarchy and Inheritance**

```
LightningModule (PyTorch Lightning)
└── PerturbationModel (Abstract Base Class)
    ├── StateTransitionPerturbationModel
    ├── EmbedSumPerturbationModel
    ├── OldNeuralOTPerturbationModel
    ├── PerturbMeanPerturbationModel
    ├── ContextMeanPerturbationModel
    ├── DecoderOnlyPerturbationModel
    ├── PseudobulkPerturbationModel
    ├── CPAPerturbationModel
    ├── SCVIPerturbationModel
    └── scGPTForPerturbation
```

### **5. StateTransitionPerturbationModel Internal Flow**

```
StateTransitionPerturbationModel
├── __init__()
│   ├── super().__init__() → PerturbationModel.__init__()
│   ├── _build_networks() → create neural network components
│   ├── ConfidenceToken (optional) → uncertainty quantification
│   └── gene_decoder (optional) → LatentToGeneDecoder
│
├── _build_networks()
│   ├── pert_encoder → MLP for perturbation encoding
│   ├── basal_encoder → MLP for control cell encoding  
│   ├── transformer_backbone → GPT-2 or LLaMA model
│   └── project_out → MLP for output projection
│
├── forward()
│   ├── encode_perturbation() → pert_encoder(pert)
│   ├── encode_basal_expression() → basal_encoder(basal)
│   ├── combined_input = pert_embedding + control_cells
│   ├── batch_encoder (optional) → add batch embeddings
│   ├── confidence_token (optional) → append confidence token
│   ├── transformer_backbone() → process sequence
│   ├── extract_confidence_prediction() (optional)
│   ├── project_out() → final projection
│   └── residual prediction (optional) → add to control
│
├── training_step()
│   ├── forward() → get predictions
│   ├── loss_fn() → compute main loss
│   ├── gene_decoder (optional) → compute decoder loss
│   ├── confidence_loss (optional) → compute confidence loss
│   └── regularization (optional) → add L1 regularization
│
├── validation_step()
│   ├── forward() → get predictions
│   ├── loss_fn() → compute validation loss
│   └── gene_decoder (optional) → compute decoder loss
│
└── predict_step()
    ├── forward() → get predictions
    └── gene_decoder (optional) → decode to gene space
```

### **6. Component Dependencies**

```
StateTransitionPerturbationModel
├── CombinedLoss (nn.Module)
│   ├── SamplesLoss("sinkhorn")
│   └── SamplesLoss("energy")
│
├── ConfidenceToken (nn.Module)
│   ├── confidence_token (Parameter)
│   └── confidence_projection (Sequential)
│
├── LatentToGeneDecoder (nn.Module)
│   ├── blocks (ModuleList) or decoder (Sequential)
│   └── final_layer (Sequential)
│
├── Transformer Backbone
│   ├── GPT2BidirectionalModel or LlamaBidirectionalModel
│   └── LoRA adapters (optional)
│
└── Utility Functions
    ├── build_mlp()
    ├── get_activation_class()
    ├── get_transformer_backbone()
    └── apply_lora()
```

### **7. Data Flow Through Training**

```
DataModule
├── train_dataloader() → batches of data
│
StateTransitionPerturbationModel.training_step()
├── batch = {"pert_emb", "ctrl_cell_emb", "pert_cell_emb", ...}
├── forward(batch) → predictions
├── loss_fn(predictions, targets) → main loss
├── gene_decoder(predictions) → gene predictions (optional)
├── confidence_token → uncertainty estimates (optional)
└── return total_loss
│
Lightning Trainer
├── backward(total_loss) → compute gradients
├── optimizer.step() → update parameters
└── validation_step() → evaluate on validation set
```

### **8. Key Design Patterns**

1. **Factory Pattern**: `get_lightning_module()` creates different model types based on configuration
2. **Template Method Pattern**: `PerturbationModel` defines the training/validation interface, subclasses implement specific logic
3. **Strategy Pattern**: Different loss functions (`CombinedLoss`, `SamplesLoss`) can be swapped
4. **Decorator Pattern**: Optional components like `ConfidenceToken` and `LatentToGeneDecoder` extend functionality
5. **Builder Pattern**: `build_mlp()` constructs neural network layers with consistent structure

### **9. Configuration Flow**

```
YAML Config Files
├── configs/config.yaml → main configuration
├── model.name → determines which model class to instantiate
├── model.kwargs → parameters passed to model constructor
├── data.kwargs → data loading parameters
└── training.* → training hyperparameters
```

This architecture provides a flexible, modular system where different perturbation modeling approaches can be easily swapped through configuration, while maintaining a consistent interface for training, validation, and inference.

```plaintext
state/__main__.py
├── get_args() → parse command line arguments
├── load_hydra_config() → load configuration from YAML files
└── main() → route to appropriate command handler
    └── case "tx" → run_tx_train(cfg)
```

```plaintext
state/_cli/_tx/_train.py
├── run_tx_train(cfg)
├── get_datamodule() → create data module
├── get_lightning_module() → instantiate model
├── get_loggers() → setup logging
├── get_checkpoint_callbacks() → setup checkpointing
└── trainer.fit() → start training
```

```plaintext
state/tx/utils/__init__.py
└── get_lightning_module(model_type, data_config, model_config, training_config, var_dims)
    ├── case "neuralot" | "pertsets" | "state" → StateTransitionPerturbationModel
    ├── case "embedsum" → EmbedSumPerturbationModel  
    ├── case "old_neuralot" → OldNeuralOTPerturbationModel
    ├── case "perturb_mean" → PerturbMeanPerturbationModel
    ├── case "context_mean" → ContextMeanPerturbationModel
    ├── case "decoder_only" → DecoderOnlyPerturbationModel
    ├── case "pseudobulk" → PseudobulkPerturbationModel
    ├── case "cpa" → CPAPerturbationModel
    ├── case "scvi" → SCVIPerturbationModel
    └── case "scgpt-*" → scGPTForPerturbation
```

```plaintext
LightningModule (PyTorch Lightning)
└── PerturbationModel (Abstract Base Class)
    ├── StateTransitionPerturbationModel
    ├── EmbedSumPerturbationModel
    ├── OldNeuralOTPerturbationModel
    ├── PerturbMeanPerturbationModel
    ├── ContextMeanPerturbationModel
    ├── DecoderOnlyPerturbationModel
    ├── PseudobulkPerturbationModel
    ├── CPAPerturbationModel
    ├── SCVIPerturbationModel
    └── scGPTForPerturbation
```

```plaintext
StateTransitionPerturbationModel
├── __init__()
│   ├── super().__init__() → PerturbationModel.__init__()
│   ├── _build_networks() → create neural network components
│   ├── ConfidenceToken (optional) → uncertainty quantification
│   └── gene_decoder (optional) → LatentToGeneDecoder
│
├── _build_networks()
│   ├── pert_encoder → MLP for perturbation encoding
│   ├── basal_encoder → MLP for control cell encoding  
│   ├── transformer_backbone → GPT-2 or LLaMA model
│   └── project_out → MLP for output projection
│
├── forward()
│   ├── encode_perturbation() → pert_encoder(pert)
│   ├── encode_basal_expression() → basal_encoder(basal)
│   ├── combined_input = pert_embedding + control_cells
│   ├── batch_encoder (optional) → add batch embeddings
│   ├── confidence_token (optional) → append confidence token
│   ├── transformer_backbone() → process sequence
│   ├── extract_confidence_prediction() (optional)
│   ├── project_out() → final projection
│   └── residual prediction (optional) → add to control
│
├── training_step()
│   ├── forward() → get predictions
│   ├── loss_fn() → compute main loss
│   ├── gene_decoder (optional) → compute decoder loss
│   ├── confidence_loss (optional) → compute confidence loss
│   └── regularization (optional) → add L1 regularization
│
├── validation_step()
│   ├── forward() → get predictions
│   ├── loss_fn() → compute validation loss
│   └── gene_decoder (optional) → compute decoder loss
│
└── predict_step()
    ├── forward() → get predictions
    └── gene_decoder (optional) → decode to gene space
```

```plaintext
StateTransitionPerturbationModel
├── CombinedLoss (nn.Module)
│   ├── SamplesLoss("sinkhorn")
│   └── SamplesLoss("energy")
│
├── ConfidenceToken (nn.Module)
│   ├── confidence_token (Parameter)
│   └── confidence_projection (Sequential)
│
├── LatentToGeneDecoder (nn.Module)
│   ├── blocks (ModuleList) or decoder (Sequential)
│   └── final_layer (Sequential)
│
├── Transformer Backbone
│   ├── GPT2BidirectionalModel or LlamaBidirectionalModel
│   └── LoRA adapters (optional)
│
└── Utility Functions
    ├── build_mlp()
    ├── get_activation_class()
    ├── get_transformer_backbone()
    └── apply_lora()
```

```plaintext
DataModule
├── train_dataloader() → batches of data
│
StateTransitionPerturbationModel.training_step()
├── batch = {"pert_emb", "ctrl_cell_emb", "pert_cell_emb", ...}
├── forward(batch) → predictions
├── loss_fn(predictions, targets) → main loss
├── gene_decoder(predictions) → gene predictions (optional)
├── confidence_token → uncertainty estimates (optional)
└── return total_loss
│
Lightning Trainer
├── backward(total_loss) → compute gradients
├── optimizer.step() → update parameters
└── validation_step() → evaluate on validation set
```

```plaintext
YAML Config Files
├── configs/config.yaml → main configuration
├── model.name → determines which model class to instantiate
├── model.kwargs → parameters passed to model constructor
├── data.kwargs → data loading parameters
└── training.* → training hyperparameters
```

