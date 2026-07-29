#Dictionaries and Definitions
#This file contains
#1. SMILES_dic
#2. id_to_token
#3. functional_groups
#4. SMILES_dic variables
#5. Training_loop variables


#1. SMILES_dic
#SMILES Dictionary based on ZINC database

SMILES_dict={'[CLS]': 0, '[SEP]': 1, '[PAD]': 2, '[END]': 3, '[MSK]': 4, '#': 5, '(': 6, ')': 7, '+': 8, '-': 9, '.': 10,
             '/': 11, '1': 12, '2': 13, '3': 14, '4': 15, '5': 16, '6': 17, '7': 18, '8': 19, '9': 20,
             '=': 21, '@': 22,'A': 23, 'B': 24, 'C': 25, 'D': 26, 'F': 27, 'G': 28,'H': 29, 'I': 30,
             'K': 31, 'L': 32, 'M': 33, 'N': 34,'O': 35, 'P': 36, 'S': 37, 'T': 38, 'V': 39, 'Y': 40,
             'Z': 41,'[': 42, '\\\\': 43, ']': 44, 'a': 45, 'b': 46, 'c': 47, 'd': 48, 'e': 49, 'g': 50,
             'i': 51, 'l': 52, 'n': 53, 'o': 54, 'r': 55, 's': 56, 't': 57, 'u': 58, 'y': 59, '%': 60,      
             '0': 61, 'p': 62, '*': 63, ':': 64 }

#2. id_to_token
# Reversed mapped SMILES_dic + code to generate it 

id_to_token = {v: k for k, v in SMILES_dict.items()}
id_to_token = {0: '[CLS]', 1: '[SEP]', 2: '[PAD]', 3: '[END]', 4: '[MSK]', 5: '#', 6: '(', 7: ')', 8: '+', 9: '-', 10: '.',
               11: '/', 12: '1', 13: '2', 14: '3', 15: '4', 16: '5', 17: '6', 18: '7', 19: '8', 20: '9', 21: '=', 22: '@',
               23: 'A', 24: 'B', 25: 'C', 26: 'D', 27: 'F', 28: 'G', 29: 'H', 30: 'I', 31: 'K', 32: 'L', 33: 'M', 34: 'N',
               35: 'O', 36: 'P', 37: 'S', 38: 'T', 39: 'V', 40: 'Y', 41: 'Z', 42: '[', 43: '\\', 44: ']', 45: 'a', 46: 'b',
               47: 'c', 48: 'd', 49: 'e', 50: 'g', 51: 'i', 52: 'l', 53: 'n', 54: 'o', 55: 'r', 56: 's', 57: 't', 58: 'u',
               59: 'y', 60: '%', 61: '0', 62: 'p', 63: '*', 64: ':'}

#3. functional_groups
# SMARTS Dictionary for functional groups
functional_groups = {

    # Carbonyl-containing
    "METHYL_AMIDE": "*-[N;D2]-[C;D3](=O)-[C;D1;H3]",
    "CARBOXYLIC_ACID": "*-C(=O)[O;D1]",
    "METHYL_ESTER": "*-C(=O)[O;D2]-[C;D1;H3]",
    "ALDEHYDE": "*-C(=O)-[C;D1]",
    "AMIDE": "*-C(=O)-[N;D1]",
    "METHYL_KETONE": "*-C(=O)-[C;D1;H3]",

    # Nitrogen-containing
    "ISOCYANATE": "*-[N;D2]=[C;D2]=[O;D1]",
    "ISOTHIOCYANATE": "*-[N;D2]=[C;D2]=[S;D1]",
    "NITRO": "*-[N;D3](=[O;D1])[O;D1]",
    "NITROSO": "*-[N;R0]=[O;D1]",
    "OXIME": "*=[N;R0]-[O;D1]",
    "IMINE": "*=[N;R0]-[C;D1;H3]",
    "AZO": "*-[N;D2]=[N;D2]-[C;D1;H3]",
    "HYDRAZINE": "*-[N;D2]=[N;D1]",
    "DIAZO": "*-[N;D2]#[N;D1]",
    "CYANO": "*-[C;D2]#[N;D1]",

    # Sulfur-containing
    "PRIMARY_SULFONAMIDE": "*-[S;D4](=[O;D1])(=[O;D1])-[N;D1]",
    "METHYL_SULFONAMIDE": "*-[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]",
    "SULFONIC_ACID": "*-[S;D4](=O)(=O)-[O;D1]",
    "SULFONATE_ESTER": "*-[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]",
    "METHYL_SULFONE": "*-[S;D4](=O)(=O)-[C;D1;H3]",
    "SULFONYL_CHLORIDE": "*-[S;D4](=O)(=O)-[Cl]",
    "SULFOXIDE": "*-[S;D3](=O)-[C;D1]",
    "THIOETHER": "*-[S;D2]-[C;D1;H3]",
    "THIOL": "*-[S;D1]",
    "THIOCARBONYL": "*=[S;D1]",

    # Other common medicinal chemistry motifs
    "HALOGEN": "*-[#9,#17,#35,#53]",
    "TERT_BUTYL": "*-[C;D4]([C;D1])([C;D1])-[C;D1]",
    "TRIFLUOROMETHYL": "*-[C;D4](F)(F)F",
    "ALKYNE": "*-[C;D2]#[C;D1;H]",
    "CYCLOPROPYL": "*-[C;D3]1-[C;D2]-[C;D2]1",

    # Small substituents
    "ETHOXY": "*-[O;D2]-[C;D2]-[C;D1;H3]",
    "METHOXY": "*-[O;D2]-[C;D1;H3]",
    "HYDROXYL": "*-[O;D1]",
    "CARBONYL_O": "*=[O;D1]",
    "PRIMARY_AMINE": "*-[N;D1]",
    "IMINE_N": "*=[N;D1]",
    "NITRILE": "*#[N;D1]"
}


#More properties definitions
PROPERTY_POSITIONS = {
    "molecular_weight_z": 1,
    "logp_z": 2,
    "num_rotatable_bonds_z": 3,
    "tpsa_z": 4,
    "h_bond_donors_z": 5,
    "h_bond_acceptors_z": 6,
}
PROPERTY_NAMES = list(PROPERTY_POSITIONS.keys())
PROPERTY_LOSS_KEYS = {
    "molecular_weight_z": "molecular_weight",
    "logp_z": "logp",
    "num_rotatable_bonds_z": "num_rotatable_bonds",
    "tpsa_z": "tpsa",
    "h_bond_donors_z": "h_bond_donors",
    "h_bond_acceptors_z": "h_bond_acceptors",
}
RAW_PROPERTY_COLUMNS = {
    "molecular_weight_z": "molecular_weight",
    "logp_z": "logp",
    "num_rotatable_bonds_z": "num_rotatable_bonds",
    "tpsa_z": "tpsa",
    "h_bond_donors_z": "h_bond_donors",
    "h_bond_acceptors_z": "h_bond_acceptors",
}
SMILES_START_POS = len(PROPERTY_NAMES) + 2


#Lists and variables

#4. SMILES_dic variables
vocab_size = len(SMILES_dict)
cls_id = SMILES_dict["[CLS]"]
end_id = SMILES_dict["[END]"]
pad_id = SMILES_dict["[PAD]"]
mask_id = SMILES_dict["[MSK]"]
sep_id = SMILES_dict["[SEP]"]

max_len = 256 #Probably need to calculate it again

#5. Training_loop variables
#Training loop variables definition

train_history = []  # Batch history for consolidation
train_total_history = []
train_smiles_history = []
train_mw_history = []
train_logp_history = []
train_num_rotatable_bonds_history = []
train_tpsa_history = []
train_h_bond_donors_history = []
train_h_bond_acceptors_history = []

val_total_history = []
val_smiles_history = []
val_mw_history = []
val_logp_history = []
val_num_rotatable_bonds_history = []
val_tpsa_history = []
val_h_bond_donors_history = []
val_h_bond_acceptors_history = []


best_val_loss = float("inf")
patience = 5
counter = 0
num_epochs = 10
start_epoch = 0
lambda_mw, lambda_logp = 0.2, 0.2