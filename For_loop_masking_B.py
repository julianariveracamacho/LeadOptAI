#Running file

#Import src py files
from src.utilities import *
from src.BRICS import *
from src.functional_groups import *
from src.masking_functions import *
from src.smiles_encoder import *  #Here change to smiles_encoder_atom_features.py
from src.atom_features import *
from src.other_functions import *
from src.atom_features import build_atom_features_from_smiles
#from src.other_functions import build_atom_encoder_tensor


#Other necessary imports
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, BRICS
from rdkit.Chem.FunctionalGroups import BuildFuncGroupHierarchy
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, IterableDataset, ConcatDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Select your masking strategy
discrete_masking_functions = [SMILES_single_mask, SMILES_block_mask, SMILES_fg_mask, SMILES_brics_mask]

# Probabilities must sum to 1 (or any proportional weights)
probabilities = [0.15, 0.15, 0.35, 0.35]

# Choose one function
discrete_mask_fn = random.choices(
    discrete_masking_functions,
    weights=probabilities,
    k=1
)[0]


#Input files
path_files_train = "/data/db6/drug_discovery_ai/data/ZINC_preprocessing_data/ZINC_train/5MB"
path_files_val = "/data/db6/drug_discovery_ai/data/ZINC_preprocessing_data/ZINC_validation/5MB"
path_files_test = "/data/db6/drug_discovery_ai/data/ZINC_preprocessing_data/ZINC_test/5MB"

#Get files

#def get_files(folder_path, max_size=None):
def get_files(folder_path, max_size=None):
    file_list_valid = []

    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)

        if os.path.isfile(full_path):
            # If no size limit, accept all files
            if max_size is None or os.path.getsize(full_path) < max_size:
                file_list_valid.append(full_path)

    return file_list_valid



files_train = get_files(path_files_train)
files_val = get_files(path_files_val)
files_test = get_files(path_files_test)

print(f"Train: {len(files_train)}")
print(f"Validation: {len(files_val)}")
print(f"Test: {len(files_test)}")


# Output path
# Path
#Output files
# Path
output_checkpoints = "/data/db6/jrivera/Masterthesis/benchmarking/benchmarking_internal/data_output/data_output_model_B/01_checkpoints_global"
output_check_epoch = "/data/db6/jrivera/Masterthesis/benchmarking/benchmarking_internal/data_output/data_output_model_B/01_checkpoints_epochs"
train_png = "/data/db6/jrivera/Masterthesis/benchmarking/benchmarking_internal/data_output/data_output_model_B/01_train_loss_png"
train_table = "/data/db6/jrivera/Masterthesis/benchmarking/benchmarking_internal/data_output/data_output_model_B/01_train_loss_table"
val_png = "/data/db6/jrivera/Masterthesis/benchmarking/benchmarking_internal/data_output/data_output_model_B/01_val_loss_png"
val_table = "/data/db6/jrivera/Masterthesis/benchmarking/benchmarking_internal/data_output/data_output_model_B/01_val_loss_table"

#Call your model
model = SmilesEncoder(
    vocab_size=vocab_size,
    pad_id=pad_id,
    cls_id=cls_id,
    sep_id=sep_id,
    end_id=end_id,
    max_seq_len=max_len
)

model = model.to(device)

# Use AdamW with gradient clipping for stability
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-5)
ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
mse_loss_fn = nn.MSELoss()



def main():
    best_val_loss = float("inf")
    start_epoch = 0
    counter = 0
    
    #Main 
    #-------------Load checkpoint if exists----------------#
    checkpoint_path = os.path.join(output_checkpoints, "best_model_patience.pth")
    print("Loading checkpoint:", checkpoint_path)
    
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint:", checkpoint_path)
    
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
        load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        if load_result.missing_keys:
            print("Checkpoint missing model keys:", load_result.missing_keys)
    
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except (ValueError, RuntimeError) as e:
                print(f"Skipping optimizer state load: {e}")
    
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        counter = checkpoint.get("counter", 0)
    
        print(f"Resuming training from epoch {start_epoch}")
    
    #---------Starting the epoch loop----------------------#
    
    for epoch in range(50): #If checkpoint available, then add start (epoch, 10)
    
    #----------------Train data set------------------------#
        print("Epoch", epoch)
        for file_id, file in enumerate(files_train):
    
            if file == '.ipynb_checkpoints':
                continue
    
            file_path = file

            if not os.path.exists(file_path):
                print(f"Missing file: {file_path}")
                continue
    
            try:
                df = pd.read_csv(file_path, low_memory=False)
                
            except Exception as e:
            
                print(f"Error reading {file_path}: {e}")
                continue
    
            #---------------Data processing-----------------#
            
            property_columns = PROPERTY_NAMES
            df = prepare_property_dataframe(df, file_path)

            if df is None:
                continue
            
            # Convert encoded SMILES from string to list
            df["encoded_smiles"] = df["encoded_smiles"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            
            #--------Final Data into X and dataloaders-------#
            x = build_encoder_tensor(df, property_columns)
            
            sample_indices = torch.arange(x.shape[0])
            dataset = TensorDataset(x, sample_indices)

            data_loader = DataLoader(dataset, batch_size=128, shuffle=True) 
            # ---------------- Training ---------------- #
            print(f"Start training for file {file_id}")
            
            model.train()
            epoch_train_loss = make_loss_totals()
                
            n_train_batches = 0
    
            for batch_id, batch in enumerate(data_loader):
            
                x = batch[0].to(device)
                indices = batch[1]
                
                #Here is the point where it should come the differnt probabilities
                discrete_mask_fn = random.choices(
                    discrete_masking_functions,
                    weights=probabilities,
                    k=1
                )[0]
                print(discrete_mask_fn)
                
                #1. Apply the selected function to your discrete values and also #2. Apply the properties masking
                masked_x, y_masked, property_labels, property_masks = (
                    apply_masking_function(x, discrete_mask_fn)
                )
                    
                #3. Check whether at least one smiles was masked
                masked_x, y_masked = ensure_masked_smiles_targets(
                    masked_x=masked_x,
                    y_masked=y_masked,
                    x=x,
                    pad_id=pad_id,
                    end_id=end_id,
                    mask_id=mask_id,
                )
                                
                # Run the model
                property_logits, smiles_logits = model(masked_x)
                
                # SMILES loss
                smiles_loss = ce_loss_fn(
                    smiles_logits[:, SMILES_START_POS:, :].reshape(-1, vocab_size),
                    y_masked[:, SMILES_START_POS:].reshape(-1).long(),
                )
                
                # Property losses
                property_losses = {}
                
                for prop_name in PROPERTY_NAMES:
                    prop_mask = property_masks[prop_name]
                    prop_labels = property_labels[prop_name]
                    prop_logits = property_logits[prop_name]
                
                    property_losses[prop_name] = (
                        mse_loss_fn(prop_logits[prop_mask], prop_labels[prop_mask])
                        if prop_mask.any()
                        else torch.tensor(0.0, device=device)
                    )
                
                # Total loss
                property_loss = sum(property_losses.values())
                total_loss = smiles_loss + property_loss

                mw_loss = property_losses["molecular_weight_z"]
                logp_loss = property_losses["logp_z"]
                num_rot_bonds_loss = property_losses["num_rotatable_bonds_z"]
                tpsa_loss = property_losses["tpsa_z"]
                h_bond_donors_loss = property_losses["h_bond_donors_z"]
                h_bond_acceptors_loss = property_losses["h_bond_acceptors_z"]

              

                if not torch.isfinite(total_loss):
                    print(f"Non-finite training loss at epoch {epoch}, file {file_id}, batch {batch_id}: {total_loss.item()}")
                    continue

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_train_loss["total"] += total_loss.item()
                epoch_train_loss["smiles"] += smiles_loss.item()
                epoch_train_loss["molecular_weight"] += mw_loss.item()
                epoch_train_loss["logp"] += logp_loss.item()
                epoch_train_loss["num_rotatable_bonds"] += num_rot_bonds_loss.item()
                epoch_train_loss["tpsa"] += tpsa_loss.item()
                epoch_train_loss["h_bond_donors"] += h_bond_donors_loss.item()
                epoch_train_loss["h_bond_acceptors"] += h_bond_acceptors_loss.item()
                n_train_batches += 1

            
            if n_train_batches == 0:
                raise RuntimeError("No training batches were processed. Check train_files and tensor shapes.")

            avg_epoch_train_loss = average_loss_totals(epoch_train_loss, n_train_batches)
            
            train_total_history.append(avg_epoch_train_loss["total"])
            train_smiles_history.append(avg_epoch_train_loss["smiles"])
            train_mw_history.append(avg_epoch_train_loss["molecular_weight"])
            train_logp_history.append(avg_epoch_train_loss["logp"])
            train_num_rotatable_bonds_history.append(avg_epoch_train_loss["num_rotatable_bonds"])
            train_tpsa_history.append(avg_epoch_train_loss["tpsa"])
            train_h_bond_donors_history.append(avg_epoch_train_loss["h_bond_donors"])
            train_h_bond_acceptors_history.append(avg_epoch_train_loss["h_bond_acceptors"])
            
            #-----------Save your training loss-----------------#
            training_loss_table = loss_table(epoch, avg_epoch_train_loss)

            file_name = os.path.basename(file)
            file_stem = os.path.splitext(file_name)[0]
            
            csv_file_name = f"Loss_file_{file_id}_{file_stem}_epoch_{epoch}.csv"
            table_dir = os.path.join(train_table, csv_file_name)
            
            training_loss_table.to_csv(table_dir, index=False)
            
            print(f"Saved training CSV: {table_dir}")
           #----------Save your plot_--------------------------#
            
            fig, axs = plt.subplots(8, 1, figsize=(7, 14), sharex=True)
            plots = [
                (train_total_history, "blue", "Total training loss"),
                (train_smiles_history, "darkorange", "SMILES loss"),
                (train_mw_history, "green", "MW loss"),
                (train_logp_history, "tomato", "LogP loss"),
                (train_num_rotatable_bonds_history, "purple", "Rotatable bonds loss"),
                (train_tpsa_history, "teal", "TPSA loss"),
                (train_h_bond_donors_history, "brown", "H-bond donors loss"),
                (train_h_bond_acceptors_history, "gray", "H-bond acceptors loss"),
            ]
            for ax, (data, color, title) in zip(axs, plots):
                ax.plot(data, color=color)
                ax.set_title(title)
                ax.set_ylabel("Loss")
            axs[-1].set_xlabel("Epoch")
            fig.suptitle(f"Training Loss History through epoch {epoch}")
            plt.tight_layout()
            
            png_file_name = "File_" + str(file_id) + "_" + file_stem + "_epoc_" + str(epoch) + '.png'
            saving_dir = os.path.join(train_png, png_file_name)
            
            plt.savefig(saving_dir)
            plt.close()       
            print(f"Saved training PNG: {saving_dir}")
    
            print(format_loss_summary(epoch, "Train", avg_epoch_train_loss))
    
            # ---------------- Validation ---------------- #
            
        for file_id, file in enumerate(files_val):

            if file == '.ipynb_checkpoints':
                continue
    
            file_path = file

            if not os.path.exists(file_path):
                print(f"Missing file: {file_path}")
                continue
    
            try:
                df = pd.read_csv(file_path, low_memory=False)
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
    
    
            #---------------Data processing-----------------#
            #print(f"Start data {file_id} processing")
            
            property_columns = PROPERTY_NAMES
            df = prepare_property_dataframe(df, file_path)

            if df is None:
                continue
            
            # Convert encoded SMILES from string to list
            df["encoded_smiles"] = df["encoded_smiles"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            
            #--------Final Data into X and dataloaders-------#
            x = build_encoder_tensor(df, property_columns)
            
            sample_indices = torch.arange(x.shape[0])
            dataset = TensorDataset(x, sample_indices)

            data_loader = DataLoader(
                dataset,
                batch_size=128,
                shuffle=True
            ) 
            #------------Start your validation loop------------#
            
            print(f"Start validation set for file {file_id}")
            
            model.eval()
            epoch_val_loss = make_loss_totals()
            n_val_batches = 0
    
            with torch.no_grad():
                for batch_id, batch in enumerate(data_loader):
                    x = batch[0].to(device)
                    indices = batch[1]
                    
                    discrete_mask_fn = random.choices(
                        discrete_masking_functions,
                        weights=probabilities,
                        k=1
                    )[0]
                    
                    print(discrete_mask_fn)
                    
                    #1. Apply the selected function to your discrete values and also #2. Apply the properties masking
                    masked_x, y_masked, property_labels, property_masks = (
                        apply_masking_function(x, discrete_mask_fn)
                    )
                        
                    #3. Check whether at least one smiles was masked
                    masked_x, y_masked = ensure_masked_smiles_targets(
                        masked_x=masked_x,
                        y_masked=y_masked,
                        x=x,
                        pad_id=pad_id,
                        end_id=end_id,
                        mask_id=mask_id,
                    )
                                    
                
                    # Run the model
                    property_logits, smiles_logits = model(masked_x)
                    
                    # SMILES loss
                    smiles_loss = ce_loss_fn(
                        smiles_logits[:, SMILES_START_POS:, :].reshape(-1, vocab_size),
                        y_masked[:, SMILES_START_POS:].reshape(-1).long(),
                    )
                    
                    # Property losses
                    property_losses = {}
                    
                    for prop_name in PROPERTY_NAMES:
                        prop_mask = property_masks[prop_name]
                        prop_labels = property_labels[prop_name]
                        prop_logits = property_logits[prop_name]
                    
                        property_losses[prop_name] = (
                            mse_loss_fn(prop_logits[prop_mask], prop_labels[prop_mask])
                            if prop_mask.any()
                            else torch.tensor(0.0, device=device)
                        )
                    
                    # Total loss
                    property_loss = sum(property_losses.values())
                    total_loss = smiles_loss + property_loss

                    mw_loss = property_losses["molecular_weight_z"]
                    logp_loss = property_losses["logp_z"]
                    num_rot_bonds_loss = property_losses["num_rotatable_bonds_z"]
                    tpsa_loss = property_losses["tpsa_z"]
                    h_bond_donors_loss = property_losses["h_bond_donors_z"]
                    h_bond_acceptors_loss = property_losses["h_bond_acceptors_z"]
    
                    if not torch.isfinite(total_loss):
                        print(f"Non-finite validation loss at epoch {epoch}, file {file_id}, batch {batch_id}: {total_loss.item()}")
                        continue

                    epoch_val_loss["total"] += total_loss.item()
                    epoch_val_loss["smiles"] += smiles_loss.item()
                    epoch_val_loss["molecular_weight"] += mw_loss.item()
                    epoch_val_loss["logp"] += logp_loss.item()
                    epoch_val_loss["num_rotatable_bonds"] += num_rot_bonds_loss.item()
                    epoch_val_loss["tpsa"] += tpsa_loss.item()
                    epoch_val_loss["h_bond_donors"] += h_bond_donors_loss.item()
                    epoch_val_loss["h_bond_acceptors"] += h_bond_acceptors_loss.item()
                    n_val_batches += 1
                    
            if n_val_batches == 0:
                raise RuntimeError("No validation batches were processed. Check files_val and tensor shapes.")
                
                
            avg_epoch_val_loss = average_loss_totals(epoch_val_loss, n_val_batches)
            
            val_total_history.append(avg_epoch_val_loss["total"])
            val_smiles_history.append(avg_epoch_val_loss["smiles"])
            val_mw_history.append(avg_epoch_val_loss["molecular_weight"])
            val_logp_history.append(avg_epoch_val_loss["logp"])
            val_num_rotatable_bonds_history.append(avg_epoch_val_loss["num_rotatable_bonds"])
            val_tpsa_history.append(avg_epoch_val_loss["tpsa"])
            val_h_bond_donors_history.append(avg_epoch_val_loss["h_bond_donors"])
            val_h_bond_acceptors_history.append(avg_epoch_val_loss["h_bond_acceptors"])
            
            #-----------Save your validation loss-----------------#
            
            validation_loss_table = loss_table(epoch, avg_epoch_val_loss)

            file_name = os.path.basename(file)
            file_stem = os.path.splitext(file_name)[0]
            
            csv_file_name = f"Loss_file_{file_id}_{file_stem}_epoch_{epoch}.csv"
            table_dir = os.path.join(val_table, csv_file_name)
            
            validation_loss_table.to_csv(table_dir, index=False)
            
            print(f"Saved validation CSV: {table_dir}")
    
            #-----------validation Plotting-----------------------#
            
            fig, axs = plt.subplots(8, 1, figsize=(7, 14), sharex=True)
            plots = [
                (val_total_history, "blue", "Total validation loss"),
                (val_smiles_history, "darkorange", "SMILES loss"),
                (val_mw_history, "green", "MW loss"),
                (val_logp_history, "tomato", "LogP loss"),
                (val_num_rotatable_bonds_history, "purple", "Rotatable bonds loss"),
                (val_tpsa_history, "teal", "TPSA loss"),
                (val_h_bond_donors_history, "brown", "H-bond donors loss"),
                (val_h_bond_acceptors_history, "gray", "H-bond acceptors loss"),
            ]
            for ax, (data, color, title) in zip(axs, plots):
                ax.plot(data, color=color)
                ax.set_title(title)
                ax.set_ylabel("Loss")
            axs[-1].set_xlabel("Epoch")
            fig.suptitle(f"Validation Loss History through epoch {epoch}")
            plt.tight_layout()
            
            png_file_name = "File_" + str(file_id) + "_" + file_stem + "_epoc_" + str(epoch) + '.png'
            saving_dir = os.path.join(val_png, png_file_name)
            
            plt.savefig(saving_dir)
            plt.close()
            print(f"Saved validation PNG: {saving_dir}")
    
            print(format_loss_summary(epoch, "Val", avg_epoch_val_loss))
    
        # ---------------- Checkpoints ---------------- #
     
        # current_val_loss = avg_epoch_val_loss["total"]

        # improved = current_val_loss < best_val_loss

        # if improved:
        #     best_val_loss = current_val_loss
        #     counter = 0
        
        #     # Save best model so far
        #     best_checkpoint_path = os.path.join(output_checkpoints, "best_model_patience.pth")
        #     torch.save({
        #         "epoch": epoch,
        #         "model_state_dict": model.state_dict(),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #         "best_val_loss": best_val_loss,
        #         "counter": counter,
        #         "rng_state": torch.get_rng_state(),
        #         "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        #     }, best_checkpoint_path)
        
        #     print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        # else:
        #     counter += 1
        #     print(f"No improvement. Early stopping counter: {counter}/{patience}")
        
        
        # ---------------- Save epoch checkpoint ---------------- #
        # Option 1: save every epoch
        
        checkpoint_path = os.path.join(output_check_epoch, f"model_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "counter": counter,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, checkpoint_path)
        
        print(f"Epoch checkpoint saved: {checkpoint_path}")
        
        
        # ---------------- Early stopping condition ---------------- #
        
        # if counter >= patience:
        #     print("Early stopping triggered.")
        #     break

    print("Training complete.")

if __name__ == "__main__":
    main()