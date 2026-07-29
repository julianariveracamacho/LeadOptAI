# Masking functions
#This code contains
#1. SMILES_single_mask
#2. SMILES_block_mask
#3. SMILES_fg_mask
#4. properties_continuous_mask

#Imports
import torch
from src.utilities import *

#1. SMILES_single_mask
#Basic core SMILES single token masking fucntion
#Code finalizedd on model v9 

def SMILES_single_mask(
    x,
    pad_id,
    mask_id,
    prob_discrete=0.35,
    ignore_index=-100,
):
    """Apply MLM masking only to discrete SMILES tokens."""
    masked_x = x.clone()

    token_values = x[:, :, 0]
    token_types = x[:, :, 1]

    # Create label tensor for discrete tokens (ignore_index for non-masked)
    masked_y = torch.full(
        token_values.shape,
        ignore_index,
        dtype=torch.long,
        device=x.device
    )

    # Discrete (SMILES) masking
    # Only discrete, non-special SMILES tokens are eligible.
    smiles_positions = (
        (token_types == 1)
        & (token_values.long() != pad_id)
        & (token_values.long() != sep_id)
        & (token_values.long() != end_id)
    )

    # Exclude [CLS], property positions, and [SEP] prefix.
    smiles_positions[:, :SMILES_START_POS] = False

    
    rand_discrete = torch.rand(token_values.shape, device=x.device)
    smiles_mask = (rand_discrete < prob_discrete) & smiles_positions

    masked_y[smiles_mask] = token_values[smiles_mask].long()
    masked_x[:, :, 0][smiles_mask] = mask_id

    # Continuous masking removed
    mw_labels = logp_labels = mw_mask = logp_mask = None
    
    return masked_x, masked_y


#2. SMILES_block_mask
#Block masking on SMILES tokens, min block len of 2 (Otherwise, same as SMILES_single_mask)
#Written if smiles too smoshortrt to start directly on the first token
#Ignore special tokens and pad

def SMILES_block_mask(
    x,
    pad_id,
    mask_id,
    min_block=2,
    max_block=10,
    ignore_index=-100,
):
    masked_x = x.clone()

    token_values = x[:, :, 0]
    token_types = x[:, :, 1]

    masked_y = torch.full(
        token_values.shape,
        ignore_index,
        dtype=torch.long,
        device=x.device,
    )
    """Apply block masking to discrete SMILES tokens only.

    Args:
        x: Tensor of shape [B, L, 2] with token values and types
        max_len: Maximum sequence length
        max_block: Maximum block length (e.g., 5)
        pad_id: Padding token ID
        mask_id: Mask token ID
        ignore_index: Label index to ignore in loss calculation
    """

    discrete_positions = (
        (token_types == 1) 
        & (token_values.long() != cls_id)
        & (token_values.long() != sep_id)
        & (token_values.long() != pad_id)
        & (token_values.long() != end_id)
        )
    discrete_positions[:, :SMILES_START_POS] = False
    #Added multiple conditions to ensure only SMILES positions are masked
    discrete_mask = torch.zeros_like(discrete_positions, dtype=torch.bool)

    B, L = token_values.shape

    for b in range(B):
        valid_pos = torch.where(discrete_positions[b])[0]

        if valid_pos.numel() == 0:
            continue

        # choose block length
        block_len = torch.randint(min_block, min(max_block, valid_pos.numel()) + 1, (1,), device=x.device,).item()

        # choose a start inside the valid SMILES-token index list
        start_idx = torch.randint(
            0,
            valid_pos.numel() - block_len + 1,
            (1,),
            device=x.device,
        ).item()

        block_positions = valid_pos[start_idx : start_idx + block_len]
        discrete_mask[b, block_positions] = True

    masked_y[discrete_mask] = token_values[discrete_mask].long()

    masked_values = masked_x[:, :, 0]
    masked_values[discrete_mask] = mask_id
    masked_x[:, :, 0] = masked_values

    return masked_x, masked_y



#3. SMILES_fg_mask
#Based on functional_groups.py preprocessing only when this masking strategy is selected
#Here to be considered, only 32 functional groups limits masking probability

def SMILES_fg_mask(
    x,
    fg_occurrences_batch,
    pad_id,
    mask_id,
    prob_fg=0.5,
    ignore_index=-100,
):
    """
    Mask complete functional-group occurrences.

    Args:
        x:
            Tensor of shape [batch_size, seq_len, 2].

        fg_occurrences_batch:
            List of length batch_size. Each element contains dictionaries:
            {
                "fg_name": str,
                "fg_id": int,
                "positions": list[int],
            }

        prob_fg:
            Probability of masking each functional-group occurrence.

    Returns:
        masked_x:
            Input tensor with selected functional-group tokens replaced
            by mask_id.

        y_masked:
            Original token IDs at masked positions and ignore_index
            elsewhere.
    """
    masked_x = x.clone()

    token_values = x[:, :, 0]
    token_types = x[:, :, 1]

    y_masked = torch.full(
        token_values.shape,
        ignore_index,
        dtype=torch.long,
        device=x.device,
    )

    batch_size, seq_len = token_values.shape

    discrete_positions = (
        (token_types == 1)
        & (token_values.long() != cls_id)
        & (token_values.long() != sep_id)
        & (token_values.long() != pad_id)
        & (token_values.long() != end_id)
    )

    for sample_idx in range(batch_size):
        fg_occurrences = fg_occurrences_batch[sample_idx]

        for occurrence in fg_occurrences:
            positions = occurrence["positions"]

            valid_positions = [
                pos
                for pos in positions
                if (
                    0 <= pos < seq_len
                    and discrete_positions[sample_idx, pos]
                )
            ]

            if not valid_positions:
                continue

            should_mask = (
                torch.rand((), device=x.device).item() < prob_fg
            )

            if not should_mask:
                continue

            valid_positions = torch.tensor(
                valid_positions,
                dtype=torch.long,
                device=x.device,
            )

            # Remove repeated positions within the occurrence.
            valid_positions = torch.unique(valid_positions)

            y_masked[sample_idx, valid_positions] = (
                token_values[sample_idx, valid_positions].long()
            )

            masked_x[sample_idx, valid_positions, 0] = mask_id

    return masked_x, y_masked
                        
# #4. SMILES_brics_mask
# #Based on the calculated BRICS, mask each fragment occurrence with a 50% probability
# def SMILES_brics_mask(
#     x,
#     brics_occurrences_batch,
#     pad_id,
#     mask_id,
#     prob_brics=0.5,
#     ignore_index=-100,
# ):
#     """Mask BRICS fragment positions for x shaped (batch, seq_len, 1) or (batch, seq_len)."""

#     if x.dim() == 2:
#         x = x.unsqueeze(-1)

#     if len(brics_occurrences_batch) != x.size(0):
#         raise ValueError(
#             "brics_occurrences_batch must contain one entry per batch sample: "
#             f"got {len(brics_occurrences_batch)} for batch size {x.size(0)}"
#         )

#     masked_x = x.clone()
#     token_values = x[:, :, 0]

#     y_masked = torch.full(
#         token_values.shape,
#         ignore_index,
#         dtype=torch.long,
#         device=x.device,
#     )

#     batch_size, seq_len = token_values.shape

#     for sample_idx, occurrences in enumerate(brics_occurrences_batch):
#         for occurrence in occurrences:
#             valid_positions = [
#                 position
#                 for position in occurrence["positions"]
#                 if (
#                     0 <= position < seq_len
#                     and token_values[sample_idx, position].long().item() != pad_id
#                 )
#             ]

#             if not valid_positions:
#                 continue

#             if torch.rand((), device=x.device).item() < prob_brics:
#                 positions = torch.tensor(
#                     valid_positions,
#                     dtype=torch.long,
#                     device=x.device,
#                 )

#                 y_masked[sample_idx, positions] = token_values[
#                     sample_idx, positions
#                 ].long()

#                 masked_x[sample_idx, positions, 0] = mask_id

#     return masked_x, y_masked

#4. properties_continuous_mask
#Continuous masking function for properties and non SMILES tokens
#Now scaled to add multiple properties, based on the smiles dic

def mask_property_values(
    masked_x,
    x,
    property_positions,
    prob_continuous=0.35,
):
    """Randomly mask continuous molecular-property values."""
    token_values = x[:, :, 0]
    
    batch_size = token_values.shape[0]
    seq_len = token_values.shape[1]

    property_labels = {}
    property_masks = {}

    for name, position in property_positions.items():
        if position < 0 or position >= seq_len:
            raise IndexError(
                f"Position {position} for property '{name}' is outside "
                f"the sequence length {seq_len}."
            )

        property_labels[name] = token_values[:, position].clone()

        property_mask = (
            torch.rand(
                batch_size,
                device=x.device,
            )
            < prob_continuous
        )

        property_masks[name] = property_mask

        # Use a neutral continuous value, not the discrete mask_id.
        masked_x[property_mask, position, 0] = 0.0

    return masked_x, property_labels, property_masks


#4. mask_atom_features
#Mask also your Atom features because otherwise the model is guided by them (The model could cheat and we dont want that)
#All positions that are not -100 (the ones that are masked), the atom features should be masked as well
def mask_atom_features(atom_features, y_masked, ignore_index=-100):
    """Prevent fixed atom descriptors from revealing masked SMILES targets."""
    masked_atom_features = atom_features.clone()
    masked_atom_features[y_masked != ignore_index] = 0.0
    return masked_atom_features
