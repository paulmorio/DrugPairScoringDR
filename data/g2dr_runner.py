"""Handy script for learning many distributed representations
of specific datasets with specific hyperparameters

Make sure to check the .utils import in g2dr_funcs when using this script
"""

from chemicalx.data import DrugCombDB, DrugbankDDI, DrugComb, OncoPolyPharmacology, TwoSides
from tqdm import tqdm
from data.g2dr_funcs import g2dr

dataset_loader = DrugCombDB()
decompositions = ["wl1", "wl2", "wl3", "sp"]
embed_dims = [8, 16, 32, 64, 128, 256]
epoch_nums = [100, 250, 500, 1000]
#embed_dims = [8, 16, 32]
#epoch_nums = [5, 10]
path = "embeddings"
gexf_folder_path = ""

for decomposition in tqdm(decompositions, "Decompositions",leave=True):
    for embedding_dimension in tqdm(embed_dims, "Embedding dimensions", leave=True):
        for num_epochs in tqdm(epoch_nums, "Epoch Nums", leave=True):
            g2dr(dataset_loader, decomposition, embedding_dimension, num_epochs, path, gexf_folder_path="")
