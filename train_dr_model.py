"""This module contains a function for loading and processing the
distributed representations for the dataset in question. Then generating
a BatchGenerator that is compatible with the modified pipeline.

The modified pipeline will then train the specified model with the
input distributed representations along with its standard inputs

Results will be recorded in a results object that is returned at the end
or saved optionally as a pkl.
"""
import os
import json
import re
import torch
import networkx as nx
from pathlib import Path

from chemicalx.data import BatchGenerator
from chemicalx import modified_dr_pipeline # note use of custom pipeline
from data.g2dr_funcs import g2dr


def pipeline_dr_model(model, 
    num_epochs, 
    batch_size,
    drug_features,
    c_features,
    triples, 
    train_size=0.5, 
    random_state=123,
    bool_context_features=False,
    bool_drug_features=False,
    bool_drug_molecules=False,
    bool_drug_dr_features=False):
    """This function takes a model and its hyperparameters and trains
    it on the triples dataset using the split and random seed specified.

    After training it produces a results object which can be used 
    for further analysis.

    Args:
        model (nn.module): a chemicalx compliant model that we have implemented
            which uses the distributed representations learned
        num_epochs (int): number of epochs to train the model
        batch_size (int): batch size for training
        drug_features (chemicalx.DrugFeatures): The augmented drug features, has to contain
            the DR values if learning a model that utilises the DR 
        c_features (chemicalx.ContextFeatures): context features
        triples (chemicalx.LabeledTriples): the Labeled triples forming the drug pair dataset
        train_size (float): value between 0.0 and 1.0 that signifies how large
            the training set should be
        random_state (int): for reproduction of results
    """
    train, test = triples.train_test_split(train_size=train_size, random_state=random_state)
    train_generator = BatchGenerator(batch_size=batch_size,
                               context_features=bool_context_features,
                               drug_features=bool_drug_features,
                               drug_molecules=bool_drug_molecules,
                               drug_dr_features=bool_drug_dr_features,
                               context_feature_set=c_features,
                               drug_feature_set=drug_features,
                               labeled_triples=train)
    test_generator = BatchGenerator(batch_size=batch_size,
                               context_features=bool_context_features,
                               drug_features=bool_drug_features,
                               drug_molecules=bool_drug_molecules,
                               drug_dr_features=bool_drug_dr_features,
                               context_feature_set=c_features,
                               drug_feature_set=drug_features,
                               labeled_triples=test)

    results = modified_dr_pipeline(
        train_batch_generator = train_generator,
        test_batch_generator = test_generator,
        model=model,
        # Data arguments
        context_features=bool_context_features,
        drug_features=bool_drug_features,
        drug_molecules=bool_drug_molecules,
        drug_dr_features=bool_drug_dr_features,
        # Training arguments
        epochs=num_epochs,
    )

    return results

def save_output_results(results, fh, save_model=False):
    """Save the results object to a specified path."""
    if save_model:
        torch.save(results.model, fh+"_model.pkl")
    fh = Path(fh)
    fh_dirname = os.path.dirname(fh)
    if not os.path.isdir(fh_dirname):
        os.makedirs(fh_dirname)
    Path(fh).write_text(
        json.dumps(
            {
                "evaluation": results.metrics,
                "losses": results.losses,
                "training_time": results.train_time,
                "evaluation_time": results.evaluation_time,
            },
            indent=2,
        )
    )

def train_dr_model(embedding_dirpath,
    embedding_dataset,
    embedding_decomposition,
    embedding_dimension,
    embedding_num_epochs,
    model,
    num_epochs,
    batch_size,
    train_size=0.5,
    random_state=123,
    save_results=True,
    save_model=False,
    results_folder="results",
    bool_context_features=False,
    bool_drug_features=False,
    bool_drug_molecules=False,
    bool_drug_dr_features=False):
    """This utility function aims to bring together
    learning/loading the distributed representations
    and feeding 
    """

    # Check if results already exists
    results_fh = f"{embedding_dataset.dataset_name}_{embedding_decomposition}_{embedding_dimension}_{embedding_num_epochs}_{type(model).__name__}_{num_epochs}__{batch_size}_{train_size}_{random_state}.json"
    results_fh = os.path.join(results_folder, results_fh)
    if os.path.exists(results_fh):
        print("# This has already been learned and evaluated, returning dict of metrics")
        with open(results_fh) as fh:
            results = json.load(fh)
            return results

    # Check the DR embedding file, if its available, load the embeddings
    # otherwise learn the desired embeddings
    dataset = embedding_dataset
    drug_features = dataset.get_drug_features()
    c_features = dataset.get_context_features()
    triples = dataset.get_labeled_triples()

    # Load and torch tensor -ize the embeddings
    drug_dr_embeddings = g2dr(embedding_dataset, 
        embedding_decomposition, 
        embedding_dimension, 
        embedding_num_epochs, 
        embedding_dirpath)
    for drug in drug_dr_embeddings.keys():
        drug_dr_embeddings[drug] = torch.FloatTensor(drug_dr_embeddings[drug])
    
    # augment the drug_features with the dr embeddings
    for drug_name in drug_features.keys():
        drug_features[drug_name]['dr'] = drug_dr_embeddings[drug_name]

    # train and evaluate the DR compatible model with the new data
    results = pipeline_dr_model(model, 
        num_epochs, 
        batch_size,
        drug_features,
        c_features,
        triples, 
        train_size=train_size, 
        random_state=random_state,
        bool_context_features=bool_context_features,
        bool_drug_features=bool_drug_features,
        bool_drug_molecules=bool_drug_molecules,
        bool_drug_dr_features=bool_drug_dr_features)

    if save_results:
        save_output_results(results, results_fh, save_model=save_model)

    return results


def train_non_dr_model(model,
    dataset,
    num_epochs,
    batch_size,
    train_size,
    random_state,
    save_results,
    save_model,
    results_folder="results",
    bool_context_features=False,
    bool_drug_features=False,
    bool_drug_molecules=False,
    bool_drug_dr_features=False):
    """This utility function trains and evaluates non dr based models

    """
    # Check if results already exists
    results_fh = f"{dataset.dataset_name}_{type(model).__name__}_{num_epochs}__{batch_size}_{train_size}_{random_state}.json"
    results_fh = os.path.join(results_folder, results_fh)
    if os.path.exists(results_fh):
        print("# This has already been learned and evaluated, returning dict of metrics")
        with open(results_fh) as fh:
            results = json.load(fh)
            return results

    drug_features = dataset.get_drug_features()
    c_features = dataset.get_context_features()
    triples = dataset.get_labeled_triples()

    results = pipeline_dr_model(model, 
        num_epochs, 
        batch_size,
        drug_features,
        c_features,
        triples, 
        train_size=train_size, 
        random_state=random_state,
        bool_context_features=bool_context_features,
        bool_drug_features=bool_drug_features,
        bool_drug_molecules=bool_drug_molecules,
        bool_drug_dr_features=bool_drug_dr_features)

    if save_results:
        save_output_results(results, results_fh, save_model=save_model)

    return results
