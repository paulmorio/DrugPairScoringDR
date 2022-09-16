"""This module contains functions for learning and saving
distributed representations of the graphs in drug pair 
datasets
"""

import os
import json
import re
import torch
import networkx as nx

from tqdm import tqdm
from data.utils import mol_to_nx # Make sure to consider this when running g2dr_runner

from geometric2dr.embedding_methods.pvdbow_trainer import InMemoryTrainer
from geometric2dr.decomposition.weisfeiler_lehman_patterns import wl_corpus 
from geometric2dr.decomposition.shortest_path_patterns import sp_corpus
import geometric2dr.embedding_methods.utils as gutils

def g2dr(dataset_loader, decomposition, embedding_dim, num_epochs, path="", gexf_folder_path="data"):
    """This function loads the dataset, decomposes the molecule graphs
    according to the given decomposition algorithm and learns distributed
    representations of the given graphs via skipgram

    Args:
        dataset_loader (chemicalx.DatasetLoader): ChemicalX datasetloader like DrugCombDB()
        decomposition (str): String name for decomposition algorithm 
            to be used ("wl1", "wl2", "wl3", "sp")
        embedding_dim (int): desired dimensionality of distributed representations
        num_epochs (int): number of epochs to train the skipgram model
    
    Returns:
        graph_embeddings (dict): dictionary of {graph_name : embedding} where the
            embedding will be a torch tensor
    """
    graph_embedding_fname = f"{dataset_loader.dataset_name}_{decomposition}_{embedding_dim}_{num_epochs}.json"
    graph_embedding_fname = os.path.join(path, graph_embedding_fname)

    # First check if the file has already been made, if so just load and return that
    if os.path.exists(graph_embedding_fname):
        print(f"Graph Embedding {graph_embedding_fname} exists. Loading this file")
        with open(graph_embedding_fname) as fh:
            graph_embedding = json.load(fh)
        return graph_embedding
    
    # Otherwise we will learn the embedding with the settings
    dataset = dataset_loader
    drug_features = dataset.get_drug_features()
    c_features = dataset.get_context_features()
    triples = dataset.get_labeled_triples()

    # Settings
    dataset_name = dataset.dataset_name
    gexf_folder = os.path.join(gexf_folder_path, dataset_name + "_gexf")

    if not os.path.isdir(gexf_folder):
        os.makedirs(gexf_folder)

    # Create dictionary of unique drugs and their nxgraphs
    nx_graph_bank = {}
    for drug_id in tqdm(drug_features.keys(), "Generating NXGraphs"):
        td_mol = drug_features[drug_id]['molecule']
        rdkit_mol = td_mol.to_molecule()
        nx_mol = mol_to_nx(rdkit_mol)
        nx_graph_bank[drug_id] = nx_mol

    # Save the set of nxgraphs as gexf files
    for drug_id in tqdm(nx_graph_bank.keys(), "Saving NXGraphs as GEXF files"):
        nx_drug = nx_graph_bank[drug_id]
        nx.write_gexf(nx_drug, path=os.path.join(gexf_folder, f"{drug_id}.gexf"))

    # Lets try making a wlcorpus based on the input decomposition argument
    if decomposition == "wl1":
        wl_depth = 1
        corpus_data_dir = gexf_folder
        graph_files = gutils.get_files(corpus_data_dir, ".gexf", max_files=0)
        corpus, vocabulary, prob_map, num_graphs, graph_map = wl_corpus(graph_files, wl_depth)
        extension = ".wld" + str(wl_depth) # Extension of the graph document
    if decomposition == "wl2":
        wl_depth = 2
        corpus_data_dir = gexf_folder
        graph_files = gutils.get_files(corpus_data_dir, ".gexf", max_files=0)
        corpus, vocabulary, prob_map, num_graphs, graph_map = wl_corpus(graph_files, wl_depth)
        extension = ".wld" + str(wl_depth) # Extension of the graph document
    if decomposition == "wl3":
        wl_depth = 3
        corpus_data_dir = gexf_folder
        graph_files = gutils.get_files(corpus_data_dir, ".gexf", max_files=0)
        corpus, vocabulary, prob_map, num_graphs, graph_map = wl_corpus(graph_files, wl_depth)
        extension = ".wld" + str(wl_depth) # Extension of the graph document
    if decomposition == "sp":
        corpus_data_dir = gexf_folder
        print(corpus_data_dir)
        graph_files = gutils.get_files(corpus_data_dir, ".gexf", max_files=0)
        corpus, vocabulary, prob_map, num_graphs, graph_map = sp_corpus(corpus_data_dir, 'atomic_num')
        extension = ".spp" # Extension of the graph document

    # Instantiate a PV-DBOW trainer to learn distributed reps directly.
    min_count_patterns = 0
    trainer = InMemoryTrainer(corpus_dir=corpus_data_dir, extension=extension, max_files=0, output_fh=graph_embedding_fname,
                      emb_dimension=embedding_dim, batch_size=999999999, epochs=num_epochs, initial_lr=3e-3,
                      min_count=min_count_patterns)
    trainer.train()

    ### EDIT: Better to load the saved embeddings with the sure mapping between the graph files
    graph_embedding = {}
    with open(graph_embedding_fname) as fh:
        drug_dr_embeddings = json.load(fh) 
        for drug_fname in drug_dr_embeddings.keys():
            drug = os.path.basename(drug_fname)
            drug = drug.replace(re.findall(".gexf.*", drug)[0], "")
            graph_embedding[drug] = drug_dr_embeddings[drug_fname]

    # Resave the graph_embeddings
    with open(graph_embedding_fname, 'w') as fh:
        json.dump(graph_embedding, fh, indent=4)

    return graph_embedding



def g2dr_no_train(dataset_loader, decomposition, embedding_dim, num_epochs, path="", gexf_folder_path="data"):
    """This function loads the dataset, decomposes the molecule graphs
    according to the given decomposition algorithm it is primarily used to study the corpus
    as part of writing the paper and has no other real utility. Users may ignore this.

    Args:
        dataset_loader (chemicalx.DatasetLoader): ChemicalX datasetloader like DrugCombDB()
        decomposition (str): String name for decomposition algorithm 
            to be used ("wl1", "wl2", "wl3", "sp")
        embedding_dim (int): desired dimensionality of distributed representations
        num_epochs (int): number of epochs to train the skipgram model
    
    Returns:
        graph_embeddings (dict): dictionary of {graph_name : embedding} where the
            embedding will be a torch tensor
    """
    graph_embedding_fname = f"{dataset_loader.dataset_name}_{decomposition}_{embedding_dim}_{num_epochs}.json"
    graph_embedding_fname = os.path.join(path, graph_embedding_fname)

    # First check if the file has already been made, if so just load and return that
    if os.path.exists(graph_embedding_fname):
        print(f"Graph Embedding {graph_embedding_fname} exists. Loading this file")
        with open(graph_embedding_fname) as fh:
            graph_embedding = json.load(fh)
        return graph_embedding
    
    # Otherwise we will learn the embedding with the settings
    dataset = dataset_loader
    drug_features = dataset.get_drug_features()
    c_features = dataset.get_context_features()
    triples = dataset.get_labeled_triples()

    # Settings
    dataset_name = dataset.dataset_name
    gexf_folder = os.path.join(gexf_folder_path, dataset_name + "_gexf")

    if not os.path.isdir(gexf_folder):
        os.makedirs(gexf_folder)

    # Create dictionary of unique drugs and their nxgraphs
    nx_graph_bank = {}
    for drug_id in tqdm(drug_features.keys(), "Generating NXGraphs"):
        td_mol = drug_features[drug_id]['molecule']
        rdkit_mol = td_mol.to_molecule()
        nx_mol = mol_to_nx(rdkit_mol)
        nx_graph_bank[drug_id] = nx_mol

    # Save the set of nxgraphs as gexf files
    for drug_id in tqdm(nx_graph_bank.keys(), "Saving NXGraphs as GEXF files"):
        nx_drug = nx_graph_bank[drug_id]
        nx.write_gexf(nx_drug, path=os.path.join(gexf_folder, f"{drug_id}.gexf"))

    # Lets try making a wlcorpus based on the input decomposition argument
    if decomposition == "wl1":
        wl_depth = 1
        corpus_data_dir = gexf_folder
        graph_files = gutils.get_files(corpus_data_dir, ".gexf", max_files=0)
        corpus, vocabulary, prob_map, num_graphs, graph_map = wl_corpus(graph_files, wl_depth)
        extension = ".wld" + str(wl_depth) # Extension of the graph document
    if decomposition == "wl2":
        wl_depth = 2
        corpus_data_dir = gexf_folder
        graph_files = gutils.get_files(corpus_data_dir, ".gexf", max_files=0)
        corpus, vocabulary, prob_map, num_graphs, graph_map = wl_corpus(graph_files, wl_depth)
        extension = ".wld" + str(wl_depth) # Extension of the graph document
    if decomposition == "wl3":
        wl_depth = 3
        corpus_data_dir = gexf_folder
        graph_files = gutils.get_files(corpus_data_dir, ".gexf", max_files=0)
        corpus, vocabulary, prob_map, num_graphs, graph_map = wl_corpus(graph_files, wl_depth)
        extension = ".wld" + str(wl_depth) # Extension of the graph document
    if decomposition == "sp":
        corpus_data_dir = gexf_folder
        print(corpus_data_dir)
        graph_files = gutils.get_files(corpus_data_dir, ".gexf", max_files=0)
        corpus, vocabulary, prob_map, num_graphs, graph_map = sp_corpus(corpus_data_dir, 'atomic_num')
        extension = ".spp" # Extension of the graph document

    return corpus, vocabulary, prob_map, num_graphs, graph_map



if __name__ == '__main__':
    from chemicalx.data import DrugCombDB, DrugbankDDI, DrugComb, OncoPolyPharmacology, TwoSides
    
    # dataset_loader = DrugCombDB()
    # dataset_loader = DrugComb()
    # dataset_loader = DrugbankDDI()
    # dataset_loader = OncoPolyPharmacology()
    dataset_loader = TwoSides()
    decomposition = "sp"
    embedding_dimension = 8
    num_epochs = 10
    path = "embeddings"

    drug_dr_embeddings = g2dr(dataset_loader, decomposition, embedding_dimension, num_epochs, path)