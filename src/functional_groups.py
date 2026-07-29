#Functional groups preprocessing
#This code contains
#1. node_to_dict
#2. decode_smiles_from_tensor
#3. SMARTS_Mol_dic
#4. map_atoms_to_smiles_characters
#5. build_occurrences_batch
#6. get_FG_occurrences_from_Mol
#7. get_fg_from_SMILES_char


#Imports
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.FunctionalGroups import BuildFuncGroupHierarchy
import torch
import torch.nn as nn

#Abkurzungen
#Mol = Molecule
#FG= Functional Groups
#SMARTS= SMILES ARBITRARY Target Specification

# You need to define the functional groups patterns from Smarts
#Now we create a dic based on the rdkit but we extract it and put it here

#1. node_to_dict
#From the RDKit call the predefined functional groups
#Remember that RDKit has a nested organised tree structure with functional groups 

def node_to_dict(node):
    return {
        "name": node.name,
        "label": node.label,
        "smarts": node.smarts,
        "children": [node_to_dict(child) for child in node.children]
    }

fg_dict = {
    root.name: node_to_dict(root)
    for root in BuildFuncGroupHierarchy()
}


#2. decode_smiles_from_tensor
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


#3. SMARTS_Mol_dic
#This function extracts the predefined SMART pattern of the functional group and makes a dictionary to be called later.
# Function goes together with the functional groups smart dictionary *
#To call
#functional_group_mols, FG_dict = SMARTS_Mol_dic(fg_dict)predefined SMARTS patterns

def SMARTS_Mol_dic(functional_groups):
    """
    Initializes everything needed for functional-group masking.

    Expects functional_groups in the format:
        {
            root_name: {
                "name": ...,
                "label": ...,
                "smarts": ...,
                "children": [...]
            }
        }

    Returns:
        functional_group_mols
        FG_dict
    """

    functional_group_mols = {}
    FG_dict = {"AN": 0}

    def process_node(node):
        fg_name = node.get("name")
        fg_smarts = node.get("smarts")

        if fg_name and fg_smarts:
            fg_mol = Chem.MolFromSmarts(fg_smarts)

            if fg_mol is not None:
                functional_group_mols[fg_name] = fg_mol

                if fg_name not in FG_dict:
                    FG_dict[fg_name] = len(FG_dict)
            else:
                print(f"Invalid SMARTS skipped: {fg_name} = {fg_smarts}")

        for child in node.get("children", []):
            process_node(child)

    for root_node in functional_groups.values():
        process_node(root_node)

    print("Functional group patterns compiled")
    print(FG_dict)

    return functional_group_mols, FG_dict

#To call
functional_group_mols, FG_dict = SMARTS_Mol_dic(fg_dict)


#4. map_atoms_to_smiles_characters
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

#5. build_occurrences_batch
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




#6. get_FG_occurrences_from_Mol
#Extract functional group occurrences, how many times does a functional group occur
#Match the graph functional groups pattern to the Smiles-based graph

# This highly depends on the SMARTS functional group dictionary that you define,32 functional groups, very limited

def get_FG_occurrences_from_Mol(
    smiles,
    functional_group_mols,
    FG_dict,
    smiles_start_pos=4,
):
    """
    Returns a list of functional-group occurrences.

    Positions are tensor positions, not raw SMILES character positions.
    """
    occurrences = []
    
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)

        atom_to_chars = (
            map_atoms_to_smiles_characters(smiles)
            if mol is not None
            else None
        )

    if mol is None:
        return []
    

    atom_to_chars = map_atoms_to_smiles_characters(smiles)
    #print(atom_to_chars)
    
    if atom_to_chars is None:
        return []

    #occurrences = []

    for fg_name, fg_mol in functional_group_mols.items():

        fg_id = FG_dict[fg_name]
    
        #print("FG:", fg_name)
        #print("FG ID:", fg_id)
    
        matches = mol.GetSubstructMatches(fg_mol)
    
        #print("matches:", matches)

        for match in matches:

            char_positions = []
            
            for atom_idx in match:
                chars = atom_to_chars.get(atom_idx)
            
                if chars is None:
                    print(f"Atom {atom_idx} missing from atom_to_chars")
                    continue
            
                char_positions.extend(chars)

            char_positions = sorted(set(char_positions))
            #print(char_positions)
            
            tensor_positions = [
                smiles_start_pos + char_pos
                for char_pos in char_positions
            ]

            if len(tensor_positions) > 0:
                occurrences.append({
                    "fg_name": fg_name,
                    "fg_id": fg_id,
                    "positions": tensor_positions
                })

    return occurrences



#7. get_fg_from_SMILES_char
#All previous functions defined together into one function

def get_fg_from_SMILES_char(
    smiles,
    functional_groups,
    smiles_start_pos=4,
):
    """
    Complete pipeline:
    1. Compile SMARTS
    2. Build atom->character mapping
    3. Extract functional-group occurrences

    Returns:
        occurrences
    """

    # Step 1
    functional_group_mols, FG_dict = get_SMARTS_Mol_dic(
        functional_groups
    )

    # Step 2 (optional pre-check)
    atom_to_chars = map_atoms_to_smiles_characters(smiles)

    if atom_to_chars is None:
        return []

    # Step 3
    occurrences = get_FG_occurrences_from_Mol(
        smiles=smiles,
        functional_group_mols=functional_group_mols,
        FG_dict=FG_dict,
        smiles_start_pos=smiles_start_pos,
    )

    return occurrences

#3. SMILES_fg_mask
#Based on functional_groups.py preprocessing only when this masking strategy is selected
#Here to be considered, only 32 functional groups limits masking probability

def SMILES_fg_mask(
    x,
    fg_occurrences_batch,
    pad_id,
    mask_id,
    prob_fg=0.15,
    ignore_index=-100,
):
    """
    Masks complete functional-group regions.

    x shape:
        [batch_size, seq_len, 2]

    x[:, :, 0]:
        token values

    x[:, :, 1]:
        token types

    fg_occurrences_batch:
        list of length batch_size
        each element is a list of dictionaries like:
        {
            "fg_name": "...",
            "fg_id": int,
            "positions": [int, int, ...]
        }

    Returns:
        masked_x, y_masked
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

    for sample_idx in range(batch_size):

        fg_occurrences = fg_occurrences_batch[sample_idx]
        
        #Functional groups prescence defined as ocurrences
        for occurrence in fg_occurrences:

            positions = occurrence["positions"]

            valid_positions = [
                pos for pos in positions
                if (
                    0 <= pos < seq_len
                    and token_types[sample_idx, pos] != 0
                    and token_values[sample_idx, pos].long() != pad_id
                )
            ]

            if len(valid_positions) == 0:
                continue

            should_mask = torch.rand(1, device=x.device).item() < prob_fg

            if should_mask:
                valid_positions = torch.tensor(
                    valid_positions,
                    dtype=torch.long,
                    device=x.device,
                )

                y_masked[sample_idx, valid_positions] = (
                    token_values[sample_idx, valid_positions].long()
                )

                masked_x[sample_idx, valid_positions, 0] = mask_id

    return masked_x, y_masked


    
#To call inside the loop
# for batch_id, batch in enumerate(data_loader):

#     x = batch[0].to(device)

#     fg_occurrences_batch = []

#     for sample_idx in range(x.shape[0]):

#         # Decode tensor back to SMILES
#         smiles = decode_smiles_from_tensor(
#             encoded_smiles=x[sample_idx, :, 0],
#             id_to_token=id_to_token,
#             pad_id=pad_id
#         )

#         # Extract functional groups
#         fg_occurrences = get_FG_occurrences_from_Mol(
#             smiles=smiles,
#             functional_group_mols=functional_group_mols,
#             FG_dict=FG_dict,
#             smiles_start_pos=4,
#         )

#         fg_occurrences_batch.append(fg_occurrences)

#     # Use fg_occurrences_batch in your masking function
#     masked_x, y_masked = SMILES_fg_mask(
#         x=x,
#         fg_occurrences_batch=fg_occurrences_batch,
#         pad_id=pad_id,
#         mask_id=mask_id,
#     )