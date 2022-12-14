"""An implementation of the DeepDrugDR model."""

import torch
from torch import nn
from torchdrug.layers import GraphConv, MaxReadout

from chemicalx.compat import PackedGraph
from chemicalx.constants import TORCHDRUG_NODE_FEATURES
from chemicalx.data import DrugPairBatch
from chemicalx.models import Model

__all__ = [
    "DeepDrugDR",
]


class DeepDrugDR(Model):
    """An implementation of the DeepDrugDR model
    """

    def __init__(
        self,
        *,
        drug_dr_channels: int,
        molecule_channels: int = TORCHDRUG_NODE_FEATURES,
        num_gcn_layers: int = 4,
        gcn_layer_hidden_size: int = 64,
        out_channels: int = 1,
        dropout_rate: float = 0.1,
    ):
        """Instantiate the DeepDrugDR model.

        :param molecule_channels: The number of molecular features.
        :param num_gcn_layers: Number of GCN layers.
        :param gcn_layer_hidden_size: number of hidden units in GCN layers
        :param out_channels: The number of output channels.
        :param dropout_rate: Dropout rate on the final fully-connected layer.
        """
        super(DeepDrugDR, self).__init__()
        self.num_gcn_layers = num_gcn_layers
        self.gcn_layer_hidden_size = gcn_layer_hidden_size
        self.graph_convolution_first = GraphConv(molecule_channels, self.gcn_layer_hidden_size, batch_norm=True)

        # add remaining GCN layers
        self.layers = torch.nn.ModuleList(
            GraphConv(self.gcn_layer_hidden_size, self.gcn_layer_hidden_size, batch_norm=True)
            for _ in range(num_gcn_layers - 1)
        )

        self.readout = MaxReadout()
        self.middle_channels = (2 * self.gcn_layer_hidden_size) + (2 * drug_dr_channels) 

        self.final = nn.Sequential(
            nn.BatchNorm1d(self.middle_channels),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.middle_channels, out_channels),
            nn.Sigmoid(),
        )

    def unpack(self, batch: DrugPairBatch):
        """Return the left drug molecules, and right drug molecules."""
        return (
            batch.drug_molecules_left,
            batch.drug_molecules_right,
            batch.drug_dr_features_left,
            batch.drug_dr_features_right
        )

    def _forward_molecules(self, molecules: PackedGraph) -> torch.FloatTensor:
        features = self.graph_convolution_first(molecules, molecules.data_dict["node_feature"])
        for layer in self.layers:
            features = layer(molecules, features)
        features = self.readout(molecules, features)
        return features

    def _combine_sides(self, left: torch.FloatTensor, 
        right: torch.FloatTensor, 
        drug_dr_features_left: torch.FloatTensor,
        drug_dr_features_right: torch.FloatTensor) -> torch.FloatTensor:
        return torch.cat([left, right, drug_dr_features_left, drug_dr_features_right], dim=1)

    def forward(self, molecules_left: PackedGraph, 
        molecules_right: PackedGraph,
        drug_dr_features_left: torch.FloatTensor,
        drug_dr_features_right: torch.FloatTensor) -> torch.FloatTensor:
        """
        Run a forward pass of the DeepDrug model.

        :param molecules_left: Batched molecules for the left side drugs.
        :param molecules_right: Batched molecules for the right side drugs.

        :return: A column vector of predicted synergy scores.
        """
        features_left = self._forward_molecules(molecules_left)
        features_right = self._forward_molecules(molecules_right)
        hidden = self._combine_sides(features_left, features_right, drug_dr_features_left, drug_dr_features_right)
        return self.final(hidden)
