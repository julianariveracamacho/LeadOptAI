#Atom features
#This file contains 
#All the things that you need for implementing the atom features embedding
#1. Hard-coded Definitions
#2. Compare the real chemistry symbols/atoms and compare it to the SMILES
#3. AtomFeatureEmbedding
#4. mask_atom_features


#Imports
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, IterableDataset, ConcatDataset
from rdkit import Chem, rdBase

from src.other_functions import *

#1. Hard-coded Definitions
ATOM_FEATURE_DIM = 4
atom_df = pd.read_csv("/data/db6/jrivera/Masterthesis/notebook/on_development/atom_smiles_features.csv")

FEATURE_COLUMNS = ["Atomic Number", "Electronegativity", "Covalent_radius_A", "Vdw_radius_A"]


SMILES_dict={'[CLS]': 0, '[SEP]': 1, '[PAD]': 2, '[END]': 3, '[MSK]': 4, '#': 5, '(': 6, ')': 7, '+': 8, '-': 9, '.': 10,
             '/': 11, '1': 12, '2': 13, '3': 14, '4': 15, '5': 16, '6': 17, '7': 18, '8': 19, '9': 20,
             '=': 21, '@': 22,'A': 23, 'B': 24, 'C': 25, 'D': 26, 'F': 27, 'G': 28,'H': 29, 'I': 30,
             'K': 31, 'L': 32, 'M': 33, 'N': 34,'O': 35, 'P': 36, 'S': 37, 'T': 38, 'V': 39, 'Y': 40,
             'Z': 41,'[': 42, '\\\\': 43, ']': 44, 'a': 45, 'b': 46, 'c': 47, 'd': 48, 'e': 49, 'g': 50,
             'i': 51, 'l': 52, 'n': 53, 'o': 54, 'r': 55, 's': 56, 't': 57, 'u': 58, 'y': 59, '%': 60,      
             '0': 61, 'p': 62, '*': 63, ':': 64 }

token_feature_table = torch.zeros(len(SMILES_dict), len(FEATURE_COLUMNS), dtype=torch.float)

symbol_lookup = atom_df.set_index("Symbol") # Build lookup from element symbol

#2. Compare the real chemistry symbols/atoms and compare it to the SMILES
for token, token_id in SMILES_dict.items():
    # lowercase aromatic atoms -> uppercase element
    symbol = token.upper() if token in ["b", "c", "n", "o", "p", "s"] else token

    if symbol in symbol_lookup.index:

        values = (
            symbol_lookup
            .loc[symbol, FEATURE_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype(float)
            .values
        )
        
        token_feature_table[token_id] = torch.tensor(
            values,
            dtype=torch.float32,
        )



def build_atom_features_from_smiles(smiles):
    """
    Builds atom features aligned to character-level encoded_smiles.
    Handles Cl, Br, and bracket atoms correctly using atom_to_chars.
    """

    mol = Chem.MolFromSmiles(smiles)
    atom_to_chars = map_atoms_to_smiles_characters(smiles)

    if mol is None or atom_to_chars is None:
        return None

    atom_features = [[0.0] * ATOM_FEATURE_DIM for _ in range(len(smiles))]

    for atom_idx, char_positions in atom_to_chars.items():
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()

        if symbol not in symbol_lookup.index:
            feature_vector = [0.0] * ATOM_FEATURE_DIM
        else:
            feature_vector = (
                symbol_lookup
                .loc[symbol, FEATURE_COLUMNS]
                .astype(float)
                .tolist()
            )

        for pos in char_positions:
            atom_features[pos] = feature_vector

    return atom_features


#3. AtomFeatureEmbedding
#nn.Module to encode the atom features that are going to be buffered in the forward pass
class AtomFeatureEmbedding(nn.Module):
    """
    Converts fixed atom features into a d_model embedding.
    """

    def __init__(
        self,
        atom_feature_dim=ATOM_FEATURE_DIM,
        d_model=512,
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                atom_feature_dim,
                d_model // 4,
                bias=False,
            ),

            nn.ReLU(),

            nn.Linear(
                d_model // 4,
                d_model // 2,
                bias=False,
            ),

            nn.ReLU(),

            nn.Linear(
                d_model // 2,
                d_model,
                bias=False,
            ),

            nn.LayerNorm(d_model),
        )

    def forward(self, atom_features):
        """
        atom_features:
            [batch, seq_len, atom_feature_dim]

        returns:
            [batch, seq_len, d_model]
        """

        return self.net(atom_features)


#4. mask_atom_features
#Mask also your Atom features because otherwise the model is guided by them (The model could cheat and we dont want that)
#All positions that are not -100 (the ones that are masked), the atom features should be masked as well
def mask_atom_features(atom_features, y_masked, ignore_index=-100):
    """Prevent fixed atom descriptors from revealing masked SMILES targets."""
    masked_atom_features = atom_features.clone()
    masked_atom_features[y_masked != ignore_index] = 0.0
    return masked_atom_features


#5.2. build_atom_encoder_tensor
#The tensor builder required for atom features
def build_atom_encoder_tensor(df, property_columns):
    """
    Builds:
        x: [B, max_len, 2]
        atom_features: [B, max_len, ATOM_FEATURE_DIM]
    """

    padded = []
    padded_atom_features = []

    for _, row in df.iterrows():

        properties = row[property_columns].tolist()
        encoded_smiles = row["encoded_smiles"]
        smiles = row["smiles"]

        smiles_atom_features = build_atom_features_from_smiles(smiles)

        if smiles_atom_features is None:
            continue

        zero_atom_feature = [0.0] * ATOM_FEATURE_DIM

        divided_seq = (
            [cls_id]
            + properties
            + [sep_id]
            + encoded_smiles
            + [end_id]
        )

        atom_feature_seq = (
            [zero_atom_feature]
            + [zero_atom_feature] * len(properties)
            + [zero_atom_feature]
            + smiles_atom_features
            + [zero_atom_feature]
        )

        divided_seq = divided_seq[:max_len]
        atom_feature_seq = atom_feature_seq[:max_len]

        pad_len = max_len - len(divided_seq)

        divided_seq += [pad_id] * pad_len
        atom_feature_seq += [zero_atom_feature] * pad_len

        padded.append(divided_seq)
        padded_atom_features.append(atom_feature_seq)

    X = torch.tensor(padded, dtype=torch.float)

    type_token = torch.ones_like(X)
    for position in PROPERTY_POSITIONS.values():
        type_token[:, position] = 0

    x = torch.stack((X, type_token), dim=2)

    atom_features = torch.tensor(
        padded_atom_features,
        dtype=torch.float,
    )
    
    return x, atom_features
    















        