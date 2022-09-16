"""An implementation of the EPGCN-DS-DR model."""

import torch
from torch import nn
from torchdrug.layers import MeanReadout
from torchdrug.models import GraphConvolutionalNetwork

from chemicalx.compat import PackedGraph
from chemicalx.constants import TORCHDRUG_NODE_FEATURES
from chemicalx.data import DrugPairBatch
from chemicalx.models import Model

__all__ = [
    "EPGCNDSDR",
]


class EPGCNDSDR(Model):
    r"""An implementation of the EPGCN-DS-DR model.
    """

    def __init__(
        self,
        *,
        molecule_channels: int = TORCHDRUG_NODE_FEATURES,
        drug_dr_channels: int,
        hidden_channels: int = 32,
        middle_channels: int = 16,
        out_channels: int = 1,
    ):
        """Instantiate the EPGCN-DS model.

        :param molecule_channels: The number of molecular features.
        :param hidden_channels: The number of graph convolutional filters.
        :param middle_channels: The number of hidden layer neurons in the last layer.
        :param out_channels: The number of output channels.
        """
        super().__init__()
        self.graph_convolution_in = GraphConvolutionalNetwork(molecule_channels, hidden_channels)
        self.graph_convolution_out = GraphConvolutionalNetwork(hidden_channels, middle_channels)
        self.readout = MeanReadout()
        self.final = nn.Sequential(nn.Linear(drug_dr_channels+drug_dr_channels+middle_channels, out_channels), nn.Sigmoid())

    def unpack(self, batch: DrugPairBatch):
        """Return the left molecular graph and right molecular graph."""
        return (
            batch.drug_molecules_left,
            batch.drug_molecules_right,
            batch.drug_dr_features_left,
            batch.drug_dr_features_right
        )

    def _forward_molecules(self, molecules: PackedGraph) -> torch.FloatTensor:
        features = self.graph_convolution_in(molecules, molecules.data_dict["node_feature"])["node_feature"]
        features = self.graph_convolution_out(molecules, features)["node_feature"]
        features = self.readout(molecules, features)
        return features

    def _combine_sides(self, left: torch.FloatTensor, right: torch.FloatTensor) -> torch.FloatTensor:
        return left + right

    def forward(self, molecules_left: PackedGraph, 
        molecules_right: PackedGraph, 
        drug_dr_features_left: torch.FloatTensor,
        drug_dr_features_right: torch.FloatTensor) -> torch.FloatTensor:
        """Run a forward pass of the EPGCN-DS model.

        :param molecules_left: Batched molecules for the left side drugs.
        :param molecules_right: Batched molecules for the right side drugs.
        :returns: A column vector of predicted synergy scores.
        """
        features_left = self._forward_molecules(molecules_left)
        features_right = self._forward_molecules(molecules_right)
        hidden = self._combine_sides(features_left, features_right)
        hidden = torch.cat([drug_dr_features_left, drug_dr_features_right, hidden], dim=1)
        return self.final(hidden)
