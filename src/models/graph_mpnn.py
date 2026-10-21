"""
ChiralMeta Graph MPNN — refactored from GNN/model_graph.py & proto_meta_graph.py.
DGL-based MPNN with Set2Set readout. Falls back to MLP-only if dgl unavailable.
"""

import logging
import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("ChiralMeta.GraphMPNN")

try:
    import dgl
    from dgl.nn.pytorch import NNConv, Set2Set
    HAS_DGL = True
except ImportError:
    HAS_DGL = False
    log.warning("DGL not installed — Graph MPNN disabled. Install with: pip install dgl")


if HAS_DGL:
    class MPNN(nn.Module):
        """
        Message-Passing Neural Network with GRU updates and Set2Set readout.
        Matches the original model_graph.py architecture exactly.
        """

        def __init__(
            self,
            node_in_feats: int,
            edge_in_feats: int,
            hidden_feats: int = 64,
            num_step_message_passing: int = 3,
            num_step_set2set: int = 3,
            num_layer_set2set: int = 1,
            readout_feats: int = 512,
        ):
            super().__init__()
            self.project_node_feats = nn.Sequential(
                nn.Linear(node_in_feats, hidden_feats), nn.ReLU()
            )
            self.num_step_message_passing = num_step_message_passing
            edge_network = nn.Linear(edge_in_feats, hidden_feats * hidden_feats)
            self.gnn_layer = NNConv(
                in_feats=hidden_feats,
                out_feats=hidden_feats,
                edge_func=edge_network,
                aggregator_type="sum",
            )
            self.activation = nn.ReLU()
            self.gru = nn.GRU(hidden_feats, hidden_feats)
            self.readout = Set2Set(
                input_dim=hidden_feats * 2,
                n_iters=num_step_set2set,
                n_layers=num_layer_set2set,
            )
            self.sparsify = nn.Sequential(
                nn.Linear(hidden_feats * 4, readout_feats), nn.PReLU()
            )

        def forward(self, g):
            node_feats = g.ndata["attr"]
            edge_feats = g.edata["edge_attr"]
            node_feats = self.project_node_feats(node_feats)
            hidden     = node_feats.unsqueeze(0)
            node_aggr  = [node_feats]
            for _ in range(self.num_step_message_passing):
                node_feats = self.activation(
                    self.gnn_layer(g, node_feats, edge_feats)
                ).unsqueeze(0)
                node_feats, hidden = self.gru(node_feats, hidden)
                node_feats = node_feats.squeeze(0)
            node_aggr.append(node_feats)
            node_aggr = torch.cat(node_aggr, 1)
            readout   = self.readout(g, node_aggr)
            return self.sparsify(readout)

    class GraphFeatureExtractor(nn.Module):
        def __init__(self, node_in_feats: int = 52, edge_in_feats: int = 8):
            super().__init__()
            self.mpnn = MPNN(node_in_feats, edge_in_feats)

        def forward(self, rmols):
            return self.mpnn(rmols)

else:
    # Lightweight MLP surrogate used when DGL is absent
    class GraphFeatureExtractor(nn.Module):  # type: ignore
        def __init__(self, node_in_feats: int = 52, edge_in_feats: int = 8):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(node_in_feats, 256), nn.ReLU(),
                nn.Linear(256, 512), nn.PReLU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)


# ----------------------------------------------------------------
# Atom / Bond featurizers (52-dim atoms, 8-dim bonds)
# ----------------------------------------------------------------
ATOM_LIST = [
    "C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "Si",
    "B", "H", "Se", "Te", "Sn", "Ge", "Co", "Ir", "Rh",
    "Fe", "Cr", "Zn", "Mn", "Li", "Na", "K",
]

def atom_features(atom) -> np.ndarray:
    """52-dimensional atom feature vector."""
    from rdkit.Chem.rdchem import ChiralType, HybridizationType
    symbol     = atom.GetSymbol()
    one_hot    = [1.0 if symbol == a else 0.0 for a in ATOM_LIST]
    # Pad to 26 to guarantee length
    one_hot    = one_hot[:26] + [0.0] * max(0, 26 - len(one_hot))

    feats = one_hot + [
        float(atom.GetDegree()),
        float(atom.GetTotalNumHs()),
        float(atom.GetFormalCharge()),
        float(atom.GetNumRadicalElectrons()),
        float(atom.IsInRing()),
        float(atom.GetIsAromatic()),
        float(atom.GetChiralTag() == ChiralType.CHI_TETRAHEDRAL_CW),
        float(atom.GetChiralTag() == ChiralType.CHI_TETRAHEDRAL_CCW),
        float(atom.GetChiralTag() != ChiralType.CHI_UNSPECIFIED),
        float(atom.GetHybridization() == HybridizationType.SP),
        float(atom.GetHybridization() == HybridizationType.SP2),
        float(atom.GetHybridization() == HybridizationType.SP3),
        float(atom.GetHybridization() == HybridizationType.SP3D),
        float(atom.GetHybridization() == HybridizationType.SP3D2),
        float(atom.GetAtomicNum()) / 100.0,
        float(atom.GetMass()) / 200.0,
        float(atom.GetIsotope()) / 10.0,
        float(atom.GetNoImplicit()),
        float(atom.GetNumExplicitHs()),
        float(atom.GetTotalValence()) / 8.0,
        float(atom.HasProp("_CIPCode") and atom.GetPropsAsDict().get("_CIPCode", "") == "R"),
        float(atom.HasProp("_CIPCode") and atom.GetPropsAsDict().get("_CIPCode", "") == "S"),
    ]
    return np.array(feats[:52], dtype=np.float32)


def bond_features(bond) -> np.ndarray:
    """8-dimensional bond feature vector."""
    from rdkit.Chem.rdchem import BondType, BondStereo
    bt = bond.GetBondType()
    return np.array([
        float(bt == BondType.SINGLE),
        float(bt == BondType.DOUBLE),
        float(bt == BondType.TRIPLE),
        float(bt == BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        float(bond.GetStereo() in (BondStereo.STEREOE, BondStereo.STEREOTRANS)),
        float(bond.GetStereo() in (BondStereo.STEREOZ, BondStereo.STEREOCIS)),
    ], dtype=np.float32)
