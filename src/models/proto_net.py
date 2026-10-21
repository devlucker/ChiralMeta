"""
ChiralMeta Prototypical Network — refactored from proto_meta.py
Fixes:
  1. No hardcoded paths
  2. CPU map_location for fully_trained_proto.pt (saved on cuda:0)
  3. Filters extra state_dict keys (feature_extractor_reagents.*)
  4. Registers PrototypicalNetworkTrainerConfig in sys.modules for pickle safety
"""

import sys
import pathlib
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("ChiralMeta.ProtoNet")

ROOT        = pathlib.Path(__file__).resolve().parent.parent.parent
MODEL_DIR   = ROOT / "saved_models"
PROTO_CKPT  = MODEL_DIR / "fully_trained_proto.pt"


# ================================================================
# Config dataclass — must match the pickled class in the checkpoint
# ================================================================

@dataclass(frozen=True)
class PrototypicalNetworkTrainerConfig:
    tasks_per_batch: int             = 5
    num_support: int                 = 512
    num_query: int                   = 64
    num_train_steps: int             = 1000
    validate_every_num_steps: int    = 50
    batch_size: int                  = 256
    learning_rate: float             = 0.0001
    clip_value: Optional[float]      = 1.0
    distance_metric: str             = "mahalanobis"


# Register in sys.modules so pickle.load resolves the class
# (the checkpoint was saved from a different module path)
_COMPAT_MODULES = [
    "proto_meta_utils",
    "proto_meta_utils_graph",
    "__main__",
]
for _mod in _COMPAT_MODULES:
    _fake = type(sys)(_mod)
    _fake.PrototypicalNetworkTrainerConfig = PrototypicalNetworkTrainerConfig
    sys.modules.setdefault(_mod, _fake)


# ================================================================
# Network architecture  (1544 → 1024 → 512)
# ================================================================

class PrototypicalNetwork(nn.Module):
    """
    MLP encoder: 1544-dim fingerprint → 512-dim embedding space.
    Architecture matches the checkpoint:
      fc.0 : Linear(1544, 1024)
      fc.1 : PReLU
      fc.2 : Dropout(0.1)   <- not stored in state_dict
      fc.3 : Linear(1024, 512)
    """

    def __init__(self, config: PrototypicalNetworkTrainerConfig):
        super().__init__()
        self.config     = config
        self.fc_out_dim = 512
        fc_in_dim       = 1544

        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, 1024),   # fc.0
            nn.PReLU(),                    # fc.1
            nn.Dropout(0.1),               # fc.2  (no params → not in state_dict)
            nn.Linear(1024, self.fc_out_dim),  # fc.3
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    # ── Mahalanobis distance prototypical forward ──────────────────────────

    def forward(self, batch) -> torch.Tensor:
        support_feat, support_labels, query_feat, query_labels = batch
        sup_emb = self.encode(support_feat)
        qry_emb = self.encode(query_feat)

        if self.config.distance_metric == "mahalanobis":
            means, precisions = self._class_means_and_precisions(sup_emb, support_labels)
            n_cls    = means.size(0)
            n_query  = qry_emb.size(0)
            repeated = qry_emb.repeat(1, n_cls).view(-1, means.size(1))
            rep_means= means.repeat(n_query, 1)
            diff     = (rep_means - repeated).view(n_query, n_cls, -1).permute(1, 0, 2)
            first_half = torch.matmul(diff, precisions)
            logits     = torch.mul(first_half, diff).sum(dim=2).transpose(1, 0) * -1
        else:
            logits = self._euclidean_forward(sup_emb, qry_emb, support_labels)
        return logits

    # ── Class prototypes (for embedding plot) ─────────────────────────────

    def get_prototypes(
        self,
        support_feat: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return class mean prototypes [n_classes, 512]."""
        means, _ = self._class_means_and_precisions(support_feat, support_labels)
        return means

    # ── Internals ──────────────────────────────────────────────────────────

    def _class_means_and_precisions(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        means, precisions = [], []
        task_cov = self._estimate_cov(features)
        for c in torch.unique(labels):
            idx = (labels == c).nonzero(as_tuple=False).squeeze(1)
            cf  = features[idx]
            means.append(cf.mean(0))
            lam = min(cf.size(0) / (cf.size(0) + 1), 0.1)
            eye = 0.1 * torch.eye(cf.size(1), device=self.device)
            precisions.append(
                torch.inverse(
                    lam * self._estimate_cov(cf)
                    + (1 - lam) * task_cov
                    + eye
                )
            )
        return torch.stack(means), torch.stack(precisions)

    @staticmethod
    def _estimate_cov(x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            x = x.unsqueeze(0)
        if x.size(0) != 1:
            x = x.t()
        factor = 1.0 / max(x.size(1) - 1, 1)
        x = x - x.mean(1, keepdim=True)
        return factor * x.matmul(x.t()).squeeze()

    @staticmethod
    def _extract_class_indices(labels: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        return (labels == cls).nonzero(as_tuple=False).reshape(-1)

    @staticmethod
    def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, labels.long())

    def _euclidean_forward(
        self, sup: torch.Tensor, qry: torch.Tensor, sup_labels: torch.Tensor
    ) -> torch.Tensor:
        prototypes = torch.stack([
            sup[self._extract_class_indices(sup_labels, c)].mean(0)
            for c in torch.unique(sup_labels)
        ])
        dist = (
            (qry.unsqueeze(1) - prototypes.unsqueeze(0)).pow(2).sum(2)
        )
        return -dist


# ================================================================
# Model loader
# ================================================================

def load_proto_model(
    ckpt_path: Optional[pathlib.Path] = None,
    device: Optional[torch.device] = None,
) -> PrototypicalNetwork:
    """
    Load fully_trained_proto.pt safely on CPU, filtering extra keys.

    The checkpoint dict contains:
      model_config     -> PrototypicalNetworkTrainerConfig (pickled)
      model_state_dict -> weights including extra feature_extractor_* keys
    """
    if device is None:
        device = torch.device("cpu")
    if ckpt_path is None:
        ckpt_path = PROTO_CKPT

    log.info(f"Loading ProtoNet checkpoint from {ckpt_path} ...")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Reconstruct config
    cfg = ckpt.get("model_config", PrototypicalNetworkTrainerConfig())
    if not isinstance(cfg, PrototypicalNetworkTrainerConfig):
        # If the pickled object is from a different class (compat),
        # rebuild from its __dict__
        try:
            kw = {k: getattr(cfg, k) for k in PrototypicalNetworkTrainerConfig.__dataclass_fields__}
            cfg = PrototypicalNetworkTrainerConfig(**kw)
        except Exception:
            cfg = PrototypicalNetworkTrainerConfig()

    model = PrototypicalNetwork(cfg)

    # Filter state dict — keep only fc.0.*, fc.1.*, fc.3.*
    raw_sd = ckpt.get("model_state_dict", {})
    allowed_prefixes = ("fc.0.", "fc.1.", "fc.3.")
    filtered = {
        k: v for k, v in raw_sd.items()
        if any(k.startswith(pfx) for pfx in allowed_prefixes)
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    log.info(f"  Loaded {len(filtered)} weight tensors.")
    if unexpected:
        log.debug(f"  Unexpected (ignored): {unexpected}")

    model.eval()
    model.to(device)
    log.info("  ProtoNet ready ✓")
    return model
