"""Federated learning package."""

from .server import FederatedServer
from .client import FederatedClient
from .aggregation import federated_averaging, weighted_averaging

__all__ = ["FederatedServer", "FederatedClient", "federated_averaging", "weighted_averaging"]
