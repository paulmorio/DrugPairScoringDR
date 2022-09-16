"""
Utilities such as model resolver
"""

from chemicalx.models import (DeepSynergy, 
    EPGCNDS, DeepDDS, MatchMaker, DeepDrug, DROnly, 
    DeepDRSynergy, EPGCNDSDR, DeepDDSDR, MatchMakerDR, DeepDrugDR
    )

def model_resolver(dataset, model_name, drug_dr_channels=None):
    """
    Utility function which given the dataset, and the string name of model
    returns an instance of the model with settings appropriate for that model
    as well as the boolean values for the data_loaders working on the dataset

    Args:
        dataset (chemicalx.dataset): an instance of a chemicalx dataset
        model_name (str): string literal to resolve model
    """

    if model_name == "DeepSynergy":
        model = DeepSynergy(context_channels=dataset.context_channels, drug_channels=dataset.drug_channels)
        bool_context_features = True
        bool_drug_features = True
        bool_drug_molecules = False
        bool_drug_dr_features = False

        return model, bool_context_features, bool_drug_features, bool_drug_molecules, bool_drug_dr_features

    elif model_name == "EPGCNDS":
        model = EPGCNDS()
        bool_context_features = True
        bool_drug_features = True
        bool_drug_molecules = True
        bool_drug_dr_features = False

        return model, bool_context_features, bool_drug_features, bool_drug_molecules, bool_drug_dr_features

    elif model_name == "DeepDDS":
        model = DeepDDS(context_channels=dataset.context_channels)
        bool_context_features = True
        bool_drug_features = True
        bool_drug_molecules = True
        bool_drug_dr_features = False

        return model, bool_context_features, bool_drug_features, bool_drug_molecules, bool_drug_dr_features

    elif model_name == "MatchMaker":
        model = MatchMaker(context_channels=dataset.context_channels, drug_channels=dataset.drug_channels)
        bool_context_features = True
        bool_drug_features = True
        bool_drug_molecules = False
        bool_drug_dr_features = False

        return model, bool_context_features, bool_drug_features, bool_drug_molecules, bool_drug_dr_features

    elif model_name == "DeepDrug":
        model = DeepDrug()
        bool_context_features = False
        bool_drug_features = True
        bool_drug_molecules = True
        bool_drug_dr_features = False

        return model, bool_context_features, bool_drug_features, bool_drug_molecules, bool_drug_dr_features

    elif model_name == "DROnly":
        model = DROnly(drug_dr_channels=drug_dr_channels)
        bool_context_features = False
        bool_drug_features = False
        bool_drug_molecules = False
        bool_drug_dr_features = True

        return model, bool_context_features, bool_drug_features, bool_drug_molecules, bool_drug_dr_features

    elif model_name == "DeepDRSynergy":
        model = DeepDRSynergy(context_channels=dataset.context_channels, drug_channels=dataset.drug_channels, drug_dr_channels=drug_dr_channels)
        bool_context_features = True
        bool_drug_features = True
        bool_drug_molecules = False
        bool_drug_dr_features = True

        return model, bool_context_features, bool_drug_features, bool_drug_molecules, bool_drug_dr_features

    elif model_name == "EPGCNDSDR":
        model = EPGCNDSDR(drug_dr_channels=drug_dr_channels)
        bool_context_features = True
        bool_drug_features = True
        bool_drug_molecules = True
        bool_drug_dr_features = True

        return model, bool_context_features, bool_drug_features, bool_drug_molecules, bool_drug_dr_features

    elif model_name == "DeepDDSDR":
        model = DeepDDSDR(context_channels=dataset.context_channels, drug_dr_channels=drug_dr_channels)
        bool_context_features = True
        bool_drug_features = True
        bool_drug_molecules = True
        bool_drug_dr_features = True

        return model, bool_context_features, bool_drug_features, bool_drug_molecules, bool_drug_dr_features

    elif model_name == "MatchMakerDR":
        model = MatchMakerDR(context_channels=dataset.context_channels, drug_channels=dataset.drug_channels, drug_dr_channels=drug_dr_channels)
        bool_context_features = True
        bool_drug_features = True
        bool_drug_molecules = False
        bool_drug_dr_features = True

        return model, bool_context_features, bool_drug_features, bool_drug_molecules, bool_drug_dr_features

    elif model_name == "DeepDrugDR":
        model = DeepDrugDR(drug_dr_channels=drug_dr_channels)
        bool_context_features = False
        bool_drug_features = True
        bool_drug_molecules = True
        bool_drug_dr_features = True

        return model, bool_context_features, bool_drug_features, bool_drug_molecules, bool_drug_dr_features

    else:
        raise ValueError(f"model_name is not recognized you put {model_name}")
