#This code contains
#1. SmilesEncoder
#2. forward()

#Import src py files
from src.utilities import *

#Imports
import os
import gc
import ast
import random
import glob
import math
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader, IterableDataset, ConcatDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path


#1. SmilesEncoder
# Model definition
class SmilesEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id,
        cls_id,
        sep_id,
        end_id,
        max_seq_len,
        d_model=512,
        n_hidden_layers=8,
        n_attention_heads=8,
        ffn_dimension=512,
        activation_function="gelu",
        hidden_dropout_prob=0.1,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)
        self.cls_id = int(cls_id)
        self.sep_id = int(sep_id)
        self.end_id = int(end_id)
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)

        # Positional encoding using frequency-based sin/cos
        positional_encoding = torch.zeros(1, max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        positional_encoding[:, :, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", positional_encoding)

        # Discrete embedding for SMILES tokens
        self.discrete_embed = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_id)
        
        # Continuous embedding for molecular properties (MW, LogP) - deeper architecture for stability
        self.property_positions = PROPERTY_POSITIONS
        
        self.continuous_embed = nn.ModuleDict({
            name: nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
            for name in self.property_positions.keys()
        })

        # self.dropout = nn.Dropout(hidden_dropout_prob)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_attention_heads,
            dim_feedforward=ffn_dimension,
            dropout=hidden_dropout_prob,
            activation=activation_function,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_hidden_layers)

        # Property prediction MLPs (deeper architecture for stability)
        
        self.property_heads = nn.ModuleDict({
            name: nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1)
        )
            for name in self.property_positions.keys()
        })

        # SMILES token prediction head
        self.head_smiles = nn.Linear(d_model, vocab_size)

    

    def forward(self, x):
        """
        x: [B, L, 2]
           x[:,:,0] = token_value
           x[:,:,1] = token_type (1=discrete, 0=continuous)
        """
        token_values = x[:, :, 0]
        token_types = x[:, :, 1]

        discrete_mask = token_types == 1
        # continuous_mask = token_types == 0

        batch_size, seq_length = token_values.shape
        token_embedding = torch.zeros(batch_size, seq_length, self.d_model, device=x.device)

        # Apply discrete embedding for SMILES tokens
        token_embedding[discrete_mask] = self.discrete_embed(token_values[discrete_mask].long())

        # Apply continuous embedding for properties
        # token_embedding[continuous_mask] = self.continuous_embed(token_values[continuous_mask].unsqueeze(-1))
        for name, position in self.property_positions.items():
            token_embedding[:, position] = self.continuous_embed[name](token_values[:, position].unsqueeze(-1))

        # Add positional encoding
        h = token_embedding + self.positional_encoding[:, :seq_length, :]

        # Create padding mask (ignore PAD tokens in attention)
        pad_mask = (token_types == 1) & (token_values.long() == self.pad_id)

        # Transformer encoding
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        # Property prediction at specific positions
        property_outputs = {
            name: self.property_heads[name](h[:, position]).squeeze(-1)
            for name, position in self.property_positions.items()
        }

        # SMILES token prediction
        discrete_logits = self.head_smiles(h)
        
        return property_outputs, discrete_logits
