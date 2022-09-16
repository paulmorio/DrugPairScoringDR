"""This script runs and saves the results of DR incorporating models, that can be
later loaded and examined in other notebooks, etc.

This will use the functionalites in train_dr_model to run models across different
hyperparameters spanning the DR learning and the model learning phases.
"""

from chemicalx.data import DrugCombDB, DrugbankDDI, DrugComb, OncoPolyPharmacology, TwoSides
from tqdm import tqdm
from train_dr_model import train_dr_model
from utils import model_resolver

# one of DROnly, DeepDRSynergy, EPGCNDSDR, DeepDDSDR, MatchMakerDR, DeepDrugDR
model_name = "DeepDRSynergy"
dataset_loader = DrugCombDB()

##
## Results saving settings
##
save_results = True
save_model = False
results_folder = "results"

##
## Settings for hyper parameters and experiment seeds
##
embedding_dirpath = "data/embeddings"
all_embedding_decompositions = ["sp", "wl3"]
all_embedding_dimensions = [32, 64]
all_embedding_num_epochs = [1000]
all_num_epochs = [50]
all_batch_sizes = [8192] # 2^16
all_train_sizes = [0.5]
all_random_states = [1, 2, 3, 4, 5]

for embedding_decomposition in tqdm(all_embedding_decompositions, "all_embedding_decompositions", leave=False):
    for embedding_dimension in tqdm(all_embedding_dimensions, "all_embedding_dimensions", leave=False):
        for embedding_num_epochs in tqdm(all_embedding_num_epochs, "all_embedding_num_epochs", leave=False):
            for num_epochs in tqdm(all_num_epochs, "all_num_epochs", leave=False):
                for batch_size in tqdm(all_batch_sizes, "all_batch_sizes", leave=False):
                    for train_size in tqdm(all_train_sizes, "all_train_sizes", leave=False):
                        for random_state in tqdm(all_random_states, "all_random_states", leave=False):

                            model, bool_context_features, bool_drug_features, bool_drug_molecules, bool_drug_dr_features = model_resolver(dataset_loader, model_name, embedding_dimension)
                            results = train_dr_model(embedding_dirpath=embedding_dirpath,
                                embedding_dataset=dataset_loader,
                                embedding_decomposition=embedding_decomposition,
                                embedding_dimension=embedding_dimension,
                                embedding_num_epochs=embedding_num_epochs,
                                model=model,
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
