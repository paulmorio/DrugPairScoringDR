"""This script runs and saves the results of non DR based models, that can be
later loaded and examined in other notebooks, etc.
"""

from chemicalx.data import DrugCombDB, DrugbankDDI, DrugComb, OncoPolyPharmacology, TwoSides
from tqdm import tqdm
from train_dr_model import train_non_dr_model
from utils import model_resolver

# one of DeepSynergy, EPGCNDS, DeepDDS, MatchMaker, DeepDrug,
# model_name = "DeepSynergy"
dataset_loader = TwoSides()

##
## Results saving settings
##
save_results = True
save_model = False
results_folder = "results"

##
## Settings for hyper parameters and experiment seeds
##
all_model_names = ["DeepSynergy", "EPGCNDS", "DeepDDS", "MatchMaker", "DeepDrug"]
all_num_epochs = [1000]
all_num_epochs = [250]
all_batch_sizes = [8192] # 2^16
all_train_sizes = [0.5]
all_random_states = [1, 2, 3, 4, 5]

for model_name in tqdm(all_model_names, "all_model_names"):
    for num_epochs in tqdm(all_num_epochs, "all_num_epochs", leave=False):
        for batch_size in tqdm(all_batch_sizes, "all_batch_sizes", leave=False):
            for train_size in tqdm(all_train_sizes, "all_train_sizes", leave=False):
                for random_state in tqdm(all_random_states, "all_random_states", leave=False):

                    model, bool_context_features, bool_drug_features, bool_drug_molecules, bool_drug_dr_features = model_resolver(dataset_loader, model_name, 0)
                    results = train_non_dr_model(
                        model=model,
                        dataset=dataset_loader,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        train_size=train_size,
                        random_state=random_state,
                        save_results=save_results,
                        save_model=save_model, 
                        results_folder=results_folder,
                        bool_context_features=bool_context_features,
                        bool_drug_features=bool_drug_features,
                        bool_drug_molecules=bool_drug_molecules,
                        bool_drug_dr_features=bool_drug_dr_features)
