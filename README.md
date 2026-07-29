# ChemForge
ChemForge: Chemistry-Guided Substructure Masking for Molecular Representation Learning

A chemistry-aware Transformer framework for molecular representation learning from SMILES.

ChemForge was developed as part of a master’s thesis investigating whether chemically informed masking strategies improve masked-token prediction compared with conventional random token masking.

The model combines:

BERT-style masked language modeling for SMILES
Continuous molecular-property conditioning
Optional atom-level physicochemical features
Functional-group masking
BRICS fragment masking
Contiguous block masking
Single-token masking

The main objective is to reconstruct masked molecular substructures while learning representations that encode both molecular syntax and chemical structure.
representations that encode both molecular syntax and chemical structure.

---
##Thesis Objective

Standard masked language models typically select tokens independently or mask arbitrary contiguous spans. For molecular strings, these masking strategies may ignore chemically meaningful substructures.

This project evaluates whether masking chemically defined regions, such as functional groups or BRICS fragments, provides a more informative pretraining objective.

The central comparison includes:

Single-token masking
Contiguous block masking
Functional-group masking
BRICS fragment masking

An additional ablation investigates whether atom-level features improve molecular representation learning.

---

##Model Overview

The model is a Transformer encoder that processes sequences containing molecular properties and SMILES tokens.

Each input sequence follows the structure:

[CLS] [PROPERTY TOKENS] [SEP] [SMILES TOKENS] [END] [PAD ...]

For the current multiproperty configuration:

[CLS]
[molecular_weight]
[logP]
[num_rotatable_bonds]
[TPSA]
[h_bond_donors]
[h_bond_acceptors]
[SEP]
[SMILES]
[END]

Depending on the preprocessing implementation, the exact property positions and SMILES start position may be configured separately.

---

Supplementary information on: 
