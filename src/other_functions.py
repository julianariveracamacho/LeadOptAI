#Other relevant functions
#This code contains
#1. decode_smiles_from_tensor
#2. build_occurences_batch
#3. ensure_masked_smiles_targets
#4. apply_masking_function
#5. build_encoder_tensor
#6. prepare_property_dataframe
#7. make_loss_totals
#8. average_loss_totals
#9. loss_table
#10. format_loss_summary

#Import src py files
from src.utilities import *
from src.functional_groups import *
from src.masking_functions import*
from src.BRICS import*
from src.functional_groups import*

#Imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader, IterableDataset, ConcatDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

from rdkit.Chem import Descriptors, BRICS
from rdkit.Chem.FunctionalGroups import BuildFuncGroupHierarchy
from rdkit.Chem import BRICS

#1. decode_smiles_from_tensor
#From your x/data, extract your encoded SMILES sequence and transform it back to SMILES
#Remember that your x has the encoded smiles, which mean s the numeric sequences of the SMILES_dict
# To call
# smiles = decode_smiles_from_tensor(token_values=x[sample_idx, :, 0], token_types=x[sample_idx, :, 1],  id_to_token=id_to_token, pad_id=SMILES_dict["[PAD]"])

def decode_smiles_from_tensor(
    token_values,
    token_types,
    id_to_token,
    pad_id,
):
    chars= []
    
    for token_id, token_type in zip(token_values, token_types):

        token_id = int(token_id.item())
        token_type = int(token_type.item())

        # Skip non-SMILES tokens
        if token_type == 0:
            continue

        if token_id == pad_id:
            break

        token = id_to_token[token_id]

        if token in ["[CLS]", "[SEP]", "[END]", "[MSK]"]:
            continue

        chars.append(token)

    return "".join(chars)


#2. build_occurences_batch
#General function to decode SMILES contained in your ready X and call it one, that could be applied to both FG and BRICs
#In extractor_fn you need to decide the corresponding get_FG/get_BRICS function
def build_occurrences_batch(
    x,
    extractor_fn,
    id_to_token,
    pad_id,
    smiles_start_pos=4,
    **extractor_kwargs,
):
    """
    Build occurrence lists for every sample in a batch.

    Args:
        x: Tensor with shape (batch_size, seq_len, features)
        extractor_fn: Function that extracts occurrences from one SMILES
        id_to_token: Dictionary mapping token IDs to SMILES tokens
        pad_id: Padding token ID
        smiles_start_pos: Offset where SMILES starts in your input tensor
        **extractor_kwargs: Extra arguments needed by the extractor

    Returns:
        occurrences_batch:
        [
            [occurrence_1, occurrence_2, ...],  # sample 1
            [occurrence_1, occurrence_2, ...],  # sample 2
            ...
        ]
    """

    occurrences_batch = []

    for sample_idx in range(x.size(0)):
        smiles = decode_smiles_from_tensor(
            token_values=x[sample_idx, :, 0],
            token_types=x[sample_idx, :, 1],
            id_to_token=id_to_token,
            pad_id=pad_id,
        )

        occurrences = extractor_fn(
            smiles=smiles,
            smiles_start_pos=smiles_start_pos,
            **extractor_kwargs,
        )

        occurrences_batch.append(occurrences)

    return occurrences_batch



#3. ensure_masked_smiles_targets
#This function is to avoid errors when FG masking function does not mask any tokens.
# In that case, it applies a single mask token to any random SMILES

def ensure_masked_smiles_targets(
    masked_x,
    y_masked,
    x,
    pad_id,
    end_id,
    mask_id,
    ignore_index=-100,
):
    """Guarantee at least one supervised SMILES target per sample.

    Functional-group matching can legitimately return no occurrences, for
    example when a SMILES was truncated. CrossEntropyLoss returns NaN when
    every label is ignore_index, so those samples receive a one-token fallback.
    """
    token_values = x[:, :, 0]
    token_types = x[:, :, 1]
    eligible = (
        (token_types == 1)
        & (token_values.long() != pad_id)
        & (token_values.long() != end_id)
    )
    eligible[:, :SMILES_START_POS] = False

    for sample_idx in range(x.size(0)):
        if (y_masked[sample_idx, SMILES_START_POS:] != ignore_index).any():
            continue

        candidates = torch.where(eligible[sample_idx])[0]
        if candidates.numel() == 0:
            continue

        selected = candidates[
            torch.randint(candidates.numel(), (1,), device=x.device)
        ].item()
        y_masked[sample_idx, selected] = token_values[
            sample_idx, selected
        ].long()
        masked_x[sample_idx, selected, 0] = mask_id

    return masked_x, y_masked
    
#4. apply_masking_function
#Tell the model how to apply each masking function:
#What to do when SMILES_fg_mask is selected
#What to do when SMILES_BRICS_mask is selected
#also apply properties masking

def apply_masking_function (x, discrete_mask_fn):
    """Apply the selected stochastic SMILES mask and property masks."""
    if discrete_mask_fn == SMILES_fg_mask:
        occurrences_batch = build_occurrences_batch(
            x=x,
            extractor_fn=get_FG_occurrences_from_Mol,
            id_to_token=id_to_token,
            pad_id=pad_id,
            smiles_start_pos=SMILES_START_POS,
            functional_group_mols=functional_group_mols,
            FG_dict=FG_dict,
        )
        masked_x, y_masked = SMILES_fg_mask(
            x=x,
            fg_occurrences_batch=occurrences_batch,
            pad_id=pad_id,
            mask_id=mask_id,
        )
    elif discrete_mask_fn == SMILES_brics_mask:
        occurrences_batch = build_occurrences_batch(
            x=x,
            extractor_fn=get_BRICS_occurrences_from_Mol,
            id_to_token=id_to_token,
            pad_id=pad_id,
            smiles_start_pos=SMILES_START_POS,
        )
        masked_x, y_masked = SMILES_brics_mask(
            x=x,
            brics_occurrences_batch=occurrences_batch,
            pad_id=pad_id,
            mask_id=mask_id,
        )
    else:
        masked_x, y_masked = discrete_mask_fn(
            x=x,
            pad_id=pad_id,
            mask_id=mask_id,
        )

    masked_x, y_masked = ensure_masked_smiles_targets(
        masked_x=masked_x,
        y_masked=y_masked,
        x=x,
        pad_id=pad_id,
        end_id=end_id,
        mask_id=mask_id,
    )
    masked_x, property_labels, property_masks = mask_property_values(
        masked_x,
        x,
        PROPERTY_POSITIONS,
    )

    return masked_x, y_masked, property_labels, property_masks


#5.1. build_encoder_tensor
#Before, it was inside the code inside the data preprocessing, now separate function 
#[CLS] [mw] [logp] [...] [SEP] [SMILES] [PAD] [PAD]
def build_encoder_tensor(df, property_columns):
    """Build [value, token_type] tensors with all properties before SMILES."""
    encoded_data = df[property_columns + ["encoded_smiles"]].values.tolist()
    padded = []

    for seq in encoded_data:
        properties = seq[:len(property_columns)]
        encoded_smiles = seq[len(property_columns)]
        divided_seq = [cls_id] + properties + [sep_id] + encoded_smiles + [end_id]
        divided_seq = divided_seq[:max_len]
        seq_padded = divided_seq + [pad_id] * (max_len - len(divided_seq))
        padded.append(seq_padded)

    X = torch.tensor(padded, dtype=torch.float)
    type_token = torch.ones_like(X)
    for position in PROPERTY_POSITIONS.values():
        type_token[:, position] = 0

    return torch.stack((X, type_token), dim=2)
    
#5.2. build_atom_encoder_tensor
#The tensor builder required for atom features
def build_atom_encoder_tensor(df, property_columns):
    """
    Build:
        x: [B, max_len, 2]
        atom_features: [B, max_len, ATOM_FEATURE_DIM]
    """
    # Local import prevents the earlier circular-import problem
    from src.atom_features import build_atom_features_from_smiles

    ATOM_FEATURE_DIM = 4

    padded = []
    padded_atom_features = []

    for _, row in df.iterrows():
        properties = row[property_columns].tolist()
        encoded_smiles = row["encoded_smiles"]
        raw_smiles = row["smiles"]

        decoded_tokens = [
            id_to_token[int(token_id)]
            for token_id in encoded_smiles
        ]

        # Use the exact SMILES representation that was encoded
        smiles = "".join(decoded_tokens)

        smiles_atom_features = build_atom_features_from_smiles(smiles)

        if smiles_atom_features is None:
            continue

        if len(smiles_atom_features) != len(encoded_smiles):
            mol = Chem.MolFromSmiles(smiles)
            number_of_atoms = (
                mol.GetNumAtoms() if mol is not None else None
            )

            raise ValueError(
                "Atom-feature/token alignment mismatch:\n"
                f"Raw SMILES: {raw_smiles}\n"
                f"Decoded SMILES: {smiles}\n"
                f"encoded_smiles length: {len(encoded_smiles)}\n"
                f"atom_features length: {len(smiles_atom_features)}\n"
                f"RDKit atom count: {number_of_atoms}"
            )

        zero_atom_feature = [0.0] * ATOM_FEATURE_DIM

        divided_seq = (
            [cls_id]
            + properties
            + [sep_id]
            + encoded_smiles
            + [end_id]
        )

        atom_feature_seq = (
            [zero_atom_feature]                      # CLS
            + [zero_atom_feature] * len(properties) # Properties
            + [zero_atom_feature]                   # SEP
            + smiles_atom_features                  # SMILES
            + [zero_atom_feature]                   # END
        )

        # Truncate independently
        divided_seq = divided_seq[:max_len]
        atom_feature_seq = atom_feature_seq[:max_len]

        # Pad independently
        sequence_pad_len = max_len - len(divided_seq)
        feature_pad_len = max_len - len(atom_feature_seq)

        divided_seq += [pad_id] * sequence_pad_len
        atom_feature_seq += [zero_atom_feature] * feature_pad_len

        assert len(divided_seq) == max_len
        assert len(atom_feature_seq) == max_len

        padded.append(divided_seq)
        padded_atom_features.append(atom_feature_seq)

    if not padded:
        raise ValueError("No valid rows were available to build the tensors")

    values = torch.tensor(padded, dtype=torch.float32)

    token_types = torch.ones_like(values)

    for position in PROPERTY_POSITIONS.values():
        token_types[:, position] = 0

    x = torch.stack((values, token_types), dim=2)

    atom_features = torch.tensor(
        padded_atom_features,
        dtype=torch.float32,
    )

    return x, atom_features
    
#5.3 build_encoder_tensor_masked
#New version of function where you can pass masked smiles as well
def build_encoder_tensor_masked(
    df,
    property_columns,
    smiles_column,
):
    """
    Build a tensor with shape [N, max_len, 2].

    Channel 0:
        Property values and token IDs.

    Channel 1:
        0 = continuous molecular property
        1 = discrete token
    """

    required_columns = property_columns + [smiles_column]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing dataframe columns: {missing_columns}"
        )

    encoded_data = df[required_columns].values.tolist()

    padded_sequences = []

    for row in encoded_data:

        properties = row[:len(property_columns)]
        encoded_smiles = row[len(property_columns)]

        sequence = (
            [cls_id]
            + properties
            + [sep_id]
            + encoded_smiles
            + [end_id]
        )

        sequence = sequence[:max_len]

        padded_sequence = sequence + (
            [pad_id] * (max_len - len(sequence))
        )

        padded_sequences.append(padded_sequence)

    values = torch.tensor(
        padded_sequences,
        dtype=torch.float32,
    )

    # By default, all positions are discrete
    token_types = torch.ones_like(values)

    # Property positions are continuous
    for position in PROPERTY_POSITIONS.values():
        token_types[:, position] = 0

    encoder_tensor = torch.stack(
        (values, token_types),
        dim=2,
    )

    return encoder_tensor

#6. prepare_property_dataframe
# Function checks if passed data frame has the values in the row corresponding to the properties of the SMILES

def prepare_property_dataframe(df, file_path):
    """Ensure all model property columns exist and are numeric.

    Existing *_z columns are used as-is. If a *_z column is missing but the
    raw descriptor column exists, create a per-file z-score from the raw values.
    """
    missing_raw_columns = []

    for z_column, raw_column in RAW_PROPERTY_COLUMNS.items():
        if z_column in df.columns:
            continue

        if raw_column not in df.columns:
            missing_raw_columns.append(raw_column)
            continue

        raw_values = pd.to_numeric(df[raw_column], errors="coerce")
        mean_value = raw_values.mean(skipna=True)
        std_value = raw_values.std(skipna=True)

        if pd.notna(std_value) and std_value > 0:
            df[z_column] = (raw_values - mean_value) / std_value
        else:
            df[z_column] = 0.0

        print(f"Created {z_column} from {raw_column} for {file_path}")

    if missing_raw_columns:
        print(
            f"Skipping {file_path}: missing property columns "
            f"{sorted(set(missing_raw_columns))}. Available columns: {list(df.columns)}"
        )
        return None

    df[PROPERTY_NAMES] = df[PROPERTY_NAMES].apply(pd.to_numeric, errors="coerce")
    df = df[np.isfinite(df[PROPERTY_NAMES]).all(axis=1)].reset_index(drop=True)

    if df.empty:
        print(f"No rows with finite properties in {file_path}")
        return None

    return df
    
#7. make_loss_totals
#Start loss values from 0 for every SMILES and properties
def make_loss_totals():
    return {"total": 0.0, "smiles": 0.0, **{key: 0.0 for key in PROPERTY_LOSS_KEYS.values()}}

#8. average_loss_totals
#Average to the totals based on the values
def average_loss_totals(loss_totals, n_batches):
    return {key: value / n_batches for key, value in loss_totals.items()}

#9. loss_table
# Make a loss table with all correct loss values to be saved
def loss_table(epoch, avg_loss):
    return pd.DataFrame({
        "epoch": [epoch],
        **{f"{key} loss": [value] for key, value in avg_loss.items()},
    })

#10. format_loss_summary
#Printing of the avg_loss values
def format_loss_summary(epoch, split_name, avg_loss):
    return (
        f"Epoch {epoch} | {split_name} - "
        f"Total: {avg_loss['total']:.4f}, "
        f"SMILES: {avg_loss['smiles']:.4f}, "
        f"MW: {avg_loss['molecular_weight']:.4f}, "
        f"LogP: {avg_loss['logp']:.4f}, "
        f"RotBonds: {avg_loss['num_rotatable_bonds']:.4f}, "
        f"TPSA: {avg_loss['tpsa']:.4f}, "
        f"HBD: {avg_loss['h_bond_donors']:.4f}, "
        f"HBA: {avg_loss['h_bond_acceptors']:.4f}"
    )



