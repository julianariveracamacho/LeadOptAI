# LeadOptAI

## Chemistry-Guided Substructure Masking for Molecular Representation Learning

LeadOptAI is a chemistry-aware Transformer framework for molecular representation learning from SMILES. It was developed as part of a master’s thesis investigating whether chemically informed masking strategies can improve masked molecular reconstruction and molecular-property learning compared with conventional random masking approaches.

The framework combines:

* BERT-style bidirectional masked language modeling for SMILES
* Continuous physicochemical-property conditioning
* Optional atom-level physicochemical features
* Single-token masking
* Contiguous block masking
* Functional-group masking
* BRICS-based fragment masking
* Joint prediction of discrete SMILES tokens and continuous molecular properties

The main objective is to reconstruct masked molecular regions while learning representations that capture **SMILES syntax, molecular connectivity, chemically meaningful substructures, and physicochemical properties**.

---

# Thesis Objective

Standard masked language models typically select individual tokens independently or mask arbitrary contiguous sequence regions. Although these approaches are effective in natural-language processing, they do not explicitly account for the chemical organization of molecular structures.

In SMILES representations, randomly selected tokens do not necessarily correspond to chemically meaningful units. A masked region may therefore contain only part of an atomic environment, bond, branch, functional group, or molecular fragment.

LeadOptAI investigates whether masking **chemically defined molecular regions** provides a more informative learning objective.

Four masking strategies are evaluated:

1. **Single-token masking**
   Individual eligible SMILES tokens are selected and replaced with `[MSK]`.

2. **Contiguous block masking**
   Consecutive SMILES tokens are masked, requiring reconstruction from the surrounding molecular context.

3. **Functional-group masking**
   Functional groups are identified using SMARTS-based substructure definitions, and complete functional-group occurrences are masked.

4. **BRICS fragment masking**
   Chemically meaningful fragmentation sites are identified using the RDKit BRICS algorithm and used to guide fragment-level masking.

An ablation study evaluates the contribution of each masking strategy by individually removing each strategy from the complete masking configuration.

A separate experiment investigates whether incorporating explicit **atom-level physicochemical features** improves molecular representation learning.

---

# Model Overview

LeadOptAI is based on a **bidirectional Transformer encoder** derived from the Transformer architecture introduced by Vaswani et al. (2017).

Unlike approaches that represent both molecular structures and molecular properties exclusively as text, LeadOptAI distinguishes between two input modalities:

* **Discrete inputs:** SMILES tokens and special tokens
* **Continuous inputs:** numerical molecular properties

This design allows physicochemical properties to preserve their numerical nature rather than being converted into textual token sequences.

The model therefore jointly learns molecular structure and physicochemical information within a shared Transformer representation.

---

# Input Representation

Each molecular input contains special tokens, continuous molecular-property values, and the corresponding SMILES representation.

Conceptually, the sequence follows:

```text
[CLS] [PROPERTY VALUES] [SEP] [SMILES TOKENS] [END] [PAD ...]
```

The multiproperty configuration incorporates six physicochemical descriptors:

* Molecular weight (MW)
* LogP
* Number of rotatable bonds
* Topological polar surface area (TPSA)
* Number of hydrogen-bond donors (HBD)
* Number of hydrogen-bond acceptors (HBA)

Molecular-property values are standardized using z-score normalization before being provided to the model.

The exact property positions and the starting position of the SMILES sequence are defined by the corresponding preprocessing configuration.

---

# Discrete and Continuous Embeddings

LeadOptAI processes discrete molecular-language tokens and continuous physicochemical values through separate embedding mechanisms.

## Discrete SMILES Tokens

SMILES tokens and special tokens are mapped to learned embedding vectors using a token-embedding layer.

The special-token vocabulary includes:

```text
[CLS]
[SEP]
[PAD]
[END]
[MSK]
```

## Continuous Molecular Properties

Numerical molecular properties are not tokenized as text.

Instead, each standardized property value is projected into the Transformer embedding space using a dedicated neural-network embedding module. This preserves the continuous nature of the descriptors while mapping them to the same embedding dimensionality used for discrete SMILES tokens.

This design enables LeadOptAI to directly learn relationships between continuous physicochemical quantities and molecular structure within a shared representation.

---

# Transformer Architecture

The main LeadOptAI configuration uses:

```text
Embedding dimension: 512
Transformer layers: 8
Attention heads: 8
Feed-forward dimension: 512
Dropout: 0.1
Positional encoding: sinusoidal
```

The complete property–SMILES sequence is processed bidirectionally, allowing each sequence position to attend to both preceding and subsequent molecular context.

Two main prediction components are used:

* A **SMILES prediction head** for masked discrete tokens
* Property-specific **regression heads** for continuous molecular descriptors

---

# Masking Strategies

During training, selected molecular information is removed from the input and the model is trained to reconstruct the missing content.

## Single-Token Masking

Individual eligible SMILES tokens are selected for masking.

The current implementation uses:

```text
p = 0.35
```

Single-token masking promotes learning of fine-grained molecular syntax, atomic identity, bond information, and local chemical context.

---

## Contiguous Block Masking

A continuous region of the SMILES sequence is masked.

Current block lengths are sampled between:

```text
Minimum block length: 2 tokens
Maximum block length: 10 tokens
```

Block masking requires the model to reconstruct larger molecular regions from the surrounding sequence context.

---

## Functional-Group Masking

Functional groups are identified at the molecular-graph level using RDKit and SMARTS-based substructure definitions.

The general workflow is:

```text
SMILES
   ↓
RDKit molecular graph
   ↓
SMARTS-based functional-group matching
   ↓
Identification of corresponding SMILES positions
   ↓
Replacement with [MSK]
```

Functional-group masking is applied with:

```text
p = 0.50
```

This strategy requires the model to reconstruct complete chemically recognizable motifs rather than arbitrary character sequences.

---

## BRICS Masking

BRICS masking uses **Breaking of Retrosynthetically Interesting Chemical Substructures (BRICS)** to identify chemically meaningful molecular fragmentation sites.

RDKit is used to identify BRICS bonds and derive fragment-level structural information. Corresponding regions of the SMILES representation are then selected for masking.

The current implementation uses:

```text
p = 0.50
```

This introduces chemically informed fragment-level learning into the masked reconstruction objective.

---

# Optional Atom-Level Features

An additional LeadOptAI configuration incorporates explicit atom-level physicochemical descriptors.

The evaluated atom features include:

* Atomic number
* Electronegativity
* Covalent radius
* van der Waals radius

Atom features are aligned with their corresponding SMILES positions. Positions representing continuous molecular properties or special tokens receive zero-valued atom-feature vectors.

This experiment evaluates whether explicit atomic information provides additional predictive value beyond the chemical information that can already be learned directly from the SMILES representation.

---

# Training Objective

LeadOptAI jointly predicts masked discrete SMILES tokens and masked continuous molecular properties.

For discrete SMILES positions, the model uses **cross-entropy loss**.

For continuous physicochemical properties, prediction error is calculated using **mean squared error (MSE)**.

The combined objective enables the model to simultaneously learn:

* Molecular syntax
* Chemical connectivity
* Local atomic environments
* Molecular substructures
* Structure–property relationships

---

# Dataset

The main experiments use molecular structures obtained from the **ZINC database**.

Molecules are filtered according to predefined physicochemical and elemental constraints before model training.

The six molecular descriptors considered are:

```text
Molecular weight
LogP
Rotatable bonds
TPSA
Hydrogen-bond donors
Hydrogen-bond acceptors
```

The selected dataset is divided into:

```text
Training set
Validation set
Test set
```

The training set is used for parameter optimization.

The validation set is used for model selection and hyperparameter evaluation.

The independent test set is used for final model evaluation and benchmarking against external molecular language models.

---

# Ablation Study

To determine the individual contribution of each masking strategy, an ablation study is performed using the complete masking configuration as the reference.

The following configurations are compared:

```text
All masking strategies
No single-token masking
No block masking
No functional-group masking
No BRICS masking
```

Removing one masking strategy at a time allows its contribution to SMILES reconstruction and molecular-property learning to be evaluated independently.

The results indicate that the masking strategies provide complementary learning signals at different structural scales, ranging from individual tokens and local bonding environments to functional motifs and larger molecular fragments.

---

# Model Evaluation

LeadOptAI is evaluated for both masked SMILES reconstruction and molecular-property prediction.

## SMILES Reconstruction

Evaluation includes:

* Cross-entropy loss
* Token-level prediction accuracy
* Masked-region reconstruction performance

## Molecular Properties

Prediction performance is evaluated separately for:

* Molecular weight
* LogP
* Rotatable bonds
* TPSA
* Hydrogen-bond donors
* Hydrogen-bond acceptors

Property-specific errors and aggregated performance across all six descriptors are analyzed.

To ensure a fair comparison between models, the test dataset uses **fixed masked regions**, ensuring that all evaluated architectures reconstruct the same missing molecular information.

---

# Benchmark Models

LeadOptAI is compared with established molecular language-model architectures, including:

* **ChemBERTa**
* **MolGPT**
* **Regression Transformer**

These models represent different approaches to molecular language modeling, including bidirectional masked modeling, autoregressive molecular generation, and property-conditioned sequence modeling.

The comparison investigates whether combining continuous molecular-property representations with chemistry-aware masking improves molecular reconstruction and structure–property learning.

---

# Lead-Optimization Perspective

LeadOptAI can also be applied as a **property-aware molecular editing framework** for lead optimization.

A possible application begins with an existing lead molecule. A selected molecular region can be masked while the remaining scaffold is retained. LeadOptAI can then reconstruct alternative molecular regions while conditioning the prediction on a desired physicochemical-property profile.

Conceptually:

```text
Existing lead molecule
        ↓
Selection of modification site
        ↓
Mask selected molecular region
        ↓
Specify desired property profile
        ↓
LeadOptAI reconstruction
        ↓
Candidate molecular variants
        ↓
Chemical and property evaluation
        ↓
Structure-based re-evaluation / re-docking
        ↓
Experimental validation
```

Rather than generating an entirely new molecule from scratch, this approach could support localized molecular modification while preserving important regions of an existing lead scaffold.

LeadOptAI should therefore be viewed as a method for **narrowing the chemical search space during lead optimization**, rather than as a replacement for downstream computational or experimental validation.

Because SMILES represents molecular connectivity but does not explicitly encode three-dimensional molecular geometry, optimized structures should subsequently be evaluated using structure-based methods such as molecular docking. This is necessary to determine whether relevant binding interactions, conformations, and binding poses are preserved following molecular modification.

---

# Repository Contents

This repository provides complementary source code associated with the master’s thesis.

It contains implementations related to:

* Molecular preprocessing
* Molecular filtering
* SMILES tokenization and encoding
* Continuous property encoding
* Transformer architecture
* Functional-group recognition
* BRICS fragmentation
* Single-token masking
* Contiguous block masking
* Functional-group masking
* BRICS masking
* Optional atom-feature embeddings
* Training and validation pipelines
* Hyperparameter configurations
* Ablation experiments
* Test-set evaluation
* Molecular reconstruction
* Inference procedures

The repository is intended to complement the methodological descriptions provided in the thesis and to support **transparency, reproducibility, and further development of LeadOptAI**.

---

# Supplementary Information

Additional implementation details that exceed the scope of the main thesis text are provided in this repository.

These include:

* Complete preprocessing procedures
* Molecular filtering criteria
* Token dictionaries
* Masking implementations
* Functional-group SMARTS definitions
* BRICS-processing functions
* Model hyperparameters
* Training scripts
* Validation procedures
* Ablation-study configurations
* Evaluation scripts
* Inference and reconstruction procedures

The source code should therefore be considered **complementary material to the master’s thesis**.

---

# Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems, 30.

