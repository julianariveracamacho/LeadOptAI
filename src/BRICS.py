#This code contains
#1. map_atoms_to_smiles_characters
#2. build_occurrences_batch
#3. get_BRICS_occurrences_from_Mol
#4. SMILES_brics_mask

#Imports
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.rdmolfiles import MolToSmiles
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, IterableDataset, ConcatDataset

from src.utilities import *

#1. map_atoms_to_smiles_characters
#Maps and aligns your functional groups with SMILES
#Convert you SMILES into MOL, make it character and position specific, put maps the decompressed atoms to the smiles string
#Before *len(methoxy_smarts) multiplied by the number of char in SMARTS patterns not in the SMILES occurrences, which means Br different than B r
def map_atoms_to_smiles_characters(smiles):
    """
    Returns a dictionary:
    atom_idx -> list of SMILES character positions
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    atom_to_chars = {}

    atom_counter = 0
    i = 0

    while i < len(smiles):

        char = smiles[i]

        # Bracket atom: [C@H], [nH+], [O-], etc.
        if char == "[":
            end = smiles.find("]", i)

            if end == -1:
                i += 1
                continue

            if atom_counter < mol.GetNumAtoms():
                atom_to_chars[atom_counter] = list(range(i, end + 1))
                atom_counter += 1

            i = end + 1

        # Two-character atoms: Cl, Br, all other elements are with []
        elif smiles[i:i+2] in ["Cl", "Br"]:

            if atom_counter < mol.GetNumAtoms():
                atom_to_chars[atom_counter] = [i, i + 1]
                atom_counter += 1

            i += 2

        # One-character atoms
        elif char in [
            "B", "C", "N", "O", "P", "S", "F", "I", "H",
            "b", "c", "n", "o", "p", "s"
        ]:

            if atom_counter < mol.GetNumAtoms():
                atom_to_chars[atom_counter] = [i]
                atom_counter += 1

            i += 1

        # Bonds, branches, ring numbers, etc.
        else:
            i += 1

    return atom_to_chars
    
#2. build_occurrences_batch (also on other functions)
# general function to decode smiles and call it once, that could be applied to both FG and BRICs
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

#3. get_BRICS_occurrences_from_Mol
#After the SMILES atoms are mapped to the atoms position, then we can extract the BRICS fragments
#First identify the Atom pairs, then get the bond between them.
#Then Fragment your molecule based on the posibility of cutting the bonds.
#Finally add the Fragments to get an output like fragment_smiles, atom_indices, and the correspondent positions

def get_BRICS_occurrences_from_Mol(
    smiles,
    smiles_start_pos=4,
):
    """
    Extract BRICS fragment occurrences from one SMILES.

    Requires previously defined:
        map_atoms_to_smiles_characters(smiles)

    Returns:
        [
            {
                "fragment": fragment_smiles,
                "atom_indices": [...],
                "positions": [...]
            },
            ...
        ]
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return []

    atom_to_chars = map_atoms_to_smiles_characters(smiles)

    if atom_to_chars is None:
        return []

    brics_bonds = list(BRICS.FindBRICSBonds(mol))

    if not brics_bonds:
        return []

    bond_indices = []

    for atom_pair, brics_labels in brics_bonds:
        atom_1, atom_2 = atom_pair
        bond = mol.GetBondBetweenAtoms(atom_1, atom_2)
        #print(bond)

        if bond is not None:
            bond_indices.append(bond.GetIdx())

    if not bond_indices:
        return []

    fragmented_mol = Chem.FragmentOnBonds(
        mol,
        bond_indices,
        addDummies=True,
    )
    
    #print(fragmented_mol)
    
    atom_fragments = Chem.GetMolFrags(
        fragmented_mol,
        asMols=False,
        sanitizeFrags=False,
    )

    fragment_mols = Chem.GetMolFrags(
        fragmented_mol,
        asMols=True,
        sanitizeFrags=True,
    )

    original_num_atoms = mol.GetNumAtoms()

    occurrences = []

    for atom_indices, fragment_mol in zip(atom_fragments, fragment_mols):
        original_atom_indices = [
            atom_idx
            for atom_idx in atom_indices
            if atom_idx < original_num_atoms
        ]

        positions = []

        for atom_idx in original_atom_indices:
            char_positions = atom_to_chars.get(atom_idx, [])
            positions.extend(char_positions)

        positions = sorted(
            set(position + smiles_start_pos for position in positions)
        )
        
        #print(positions)
        
        if not positions:
            continue

        fragment_smiles = Chem.MolToSmiles(fragment_mol)
        
        #print(fragment_smiles)
        
        occurrences.append(
            {
                "fragment": fragment_smiles,
                "atom_indices": original_atom_indices,
                "positions": positions,
            }
        )

    return occurrences



#4. SMILES_brics_mask
#Based on the calculated BRICS, mask each fragment occurrence with a 50% probability
def SMILES_brics_mask(
    x,
    brics_occurrences_batch,
    pad_id,
    mask_id,
    prob_brics=0.5,
    ignore_index=-100,
):
    """Mask BRICS fragment positions for x shaped (batch, seq_len, 1) or (batch, seq_len)."""

    if x.dim() == 2:
        x = x.unsqueeze(-1)

    if len(brics_occurrences_batch) != x.size(0):
        raise ValueError(
            "brics_occurrences_batch must contain one entry per batch sample: "
            f"got {len(brics_occurrences_batch)} for batch size {x.size(0)}"
        )

   
    masked_x = x.clone()

    token_values = x[:, :, 0]
    token_types = x[:, :, 1]

    batch_size, seq_len = token_values.shape

    if len(brics_occurrences_batch) != batch_size:
        raise ValueError(
            "brics_occurrences_batch must contain one entry per batch sample: "
            f"got {len(brics_occurrences_batch)} for batch size {batch_size}"
        )

    y_masked = torch.full(
        token_values.shape,
        ignore_index,
        dtype=torch.long,
        device=x.device,
    )

    discrete_positions = (
        (token_types == 1)
        & (token_values.long() != cls_id)
        & (token_values.long() != sep_id)
        & (token_values.long() != pad_id)
        & (token_values.long() != end_id)
    )

    for sample_idx, occurrences in enumerate(brics_occurrences_batch):
        for occurrence in occurrences:
            valid_positions = [
                position
                for position in occurrence.get("positions", [])
                if (
                    0 <= position < seq_len
                    and discrete_positions[sample_idx, position].item()
                )
            ]

            if not valid_positions:
                continue

            if torch.rand((), device=x.device) < prob_brics:
                positions = torch.tensor(
                    valid_positions,
                    dtype=torch.long,
                    device=x.device,
                )

                y_masked[sample_idx, positions] = token_values[
                    sample_idx,
                    positions,
                ].long()

                masked_x[sample_idx, positions, 0] = mask_id

    return masked_x, y_masked