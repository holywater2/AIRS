import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pydantic.typing import Literal
from torch_geometric.nn import Linear, MessagePassing, global_mean_pool
from torch_geometric.nn.models.schnet import ShiftedSoftplus

from models.base import BaseSettings
from models.transformer import TransformerConv

from models.utils import RBFExpansion, Dense, ResidualLayer

class PotNetConfig(BaseSettings):
    name: Literal["potnet"]
    conv_layers: int = 3
    atom_input_features: int = 92
    inf_edge_features: int = 64
    fc_features: int = 256
    output_dim: int = 256
    output_features: int = 1
    rbf_min = -4.0
    rbf_max = 4.0
    potentials = []
    charge_map = False
    transformer = False

    class Config:
        """Configure model settings behavior."""
        env_prefix = "jv_model"




class PotNetConv(MessagePassing):

    def __init__(self, fc_features):
        super(PotNetConv, self).__init__(node_dim=0)
        self.bn = nn.BatchNorm1d(fc_features)
        self.bn_interaction = nn.BatchNorm1d(fc_features)
        self.nonlinear_full = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features)
        )
        self.nonlinear = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features),
        )

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0))
        )

        return F.relu(x + self.bn(out)) # Residual part

    def message(self, x_i, x_j, edge_attr, index):
        score = torch.sigmoid(self.bn_interaction(self.nonlinear_full(torch.cat((x_i, x_j, edge_attr), dim=1))))
        return score * self.nonlinear(torch.cat((x_i, x_j, edge_attr), dim=1))

#FIXME
class EwawldEmbedding(nn.Module):
    def __init__(self,
                 emb_size_atom: int,
                 num_hidden: int) -> None:
        super().__init__()
        
        
    def get_mlp(self, units_in, units, num_hidden, activation):
        dense1 = Dense(units_in, units, activation=activation, bias=False)
        mlp = [dense1]
        res = [
            ResidualLayer(units, nLayers=2, activation=activation)
            for i in range(num_hidden)
        ]
        mlp += res
        return torch.nn.ModuleList(mlp)
    

class PotNet_Ewald(nn.Module):

    def __init__(self, config: PotNetConfig = PotNetConfig(name="potnet")):
        super().__init__()
        self.config = config
        if not config.charge_map:
            self.atom_embedding = nn.Linear(
                config.atom_input_features, config.fc_features
            )
        else:
            self.atom_embedding = nn.Linear(
                config.atom_input_features + 10, config.fc_features
            )

        # self.infinite = True

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=config.rbf_min,
                vmax=config.rbf_max,
                bins=config.fc_features,
            ),
            nn.Linear(config.fc_features, config.fc_features),
            nn.SiLU(),
        )

        self.inf_edge_embedding = RBFExpansion(
            vmin=config.rbf_min,
            vmax=config.rbf_max,
            bins=config.inf_edge_features,
            type='multiquadric'
        )

        self.infinite_linear = nn.Linear(config.inf_edge_features, config.fc_features)

        self.infinite_bn = nn.BatchNorm1d(config.fc_features)

        self.conv_layers = nn.ModuleList(
            [
                PotNetConv(config.fc_features)
                for _ in range(config.conv_layers)
            ]
        )

        self.fc = nn.Sequential(
            nn.Linear(config.fc_features, config.fc_features), ShiftedSoftplus()
        )

        self.fc_out = nn.Linear(config.output_dim, config.output_features)

    def forward(self, data, print_data=False):
        """CGCNN function mapping graph to outputs."""
        # fixed edge features: RBF-expanded bondlengths
        edge_index = data.edge_index
        edge_features = self.edge_embedding(-0.75 / data.edge_attr)

        # inf_edge_index = data.inf_edge_index
        # inf_feat = sum([data.inf_edge_attr[:, i] * pot for i, pot in enumerate(self.config.potentials)])
        # inf_edge_features = self.inf_edge_embedding(inf_feat)
        # inf_edge_features = self.infinite_bn(F.softplus(self.infinite_linear(inf_edge_features)))

        # initial node features: atom feature network...
        if self.config.charge_map:
            node_features = self.atom_embedding(torch.cat([data.x, data.g_feats], -1))
        else:
            node_features = self.atom_embedding(data.x)

        # if not self.config.transformer:
        #     edge_index = torch.cat([data.edge_index, inf_edge_index], 1)
        #     edge_features = torch.cat([edge_features, inf_edge_features], 0)

        for i in range(self.config.conv_layers):
                node_features = self.conv_layers[i](node_features, edge_index, edge_features)

        features = global_mean_pool(node_features, data.batch)
        features = self.fc(features)
        return torch.squeeze(self.fc_out(features))
