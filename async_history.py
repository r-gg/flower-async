"""Async History module for Flower async FL.

A wrapper around the Flower History class that offers centralized and distributed
metrics per timestamp instead of per round. It also groups distributed_fit metrics
per client instead of per round.

The latest Flower (1.25.0+) History class uses server_round (int) as the key,
but for asynchronous FL we need to track timestamps. This class extends History
to support both paradigms.

Attributes:
    losses_centralized: List of (server_round, value) tuples
    metrics_centralized: Dict mapping metric names to list of (server_round, value) tuples
    metrics_distributed: Dict mapping metric names to list of (server_round, value) tuples
    metrics_distributed_fit_async: Async metrics grouped by client:
        {
            "accuracy": {
                cid1: [(timestamp1, value1), (timestamp2, value2), ...],
                ...
            }
            ...
        }
    metrics_centralized_async: Async metrics with timestamps:
        {"accuracy": [(timestamp1, value1), ...]}
    losses_centralized_async: List of (timestamp, loss) tuples for async tracking
"""

from typing import Dict

from flwr.common.typing import Scalar
from flwr.server.history import History


class AsyncHistory(History):
    """Extended History class for asynchronous federated learning.

    This class extends the base Flower History class to support timestamp-based
    metrics tracking required for asynchronous federated learning scenarios.
    """

    def __init__(self) -> None:
        """Initialize AsyncHistory with async-specific tracking structures."""
        super().__init__()
        # Async-specific tracking structures using timestamps
        self.metrics_distributed_fit_async: Dict[str, Dict[str, list]] = {}
        self.metrics_centralized_async: Dict[str, list] = {}
        self.losses_centralized_async: list = []

    def add_metrics_distributed_fit_async(
        self, client_id: str, metrics: Dict[str, Scalar], timestamp: float
    ) -> None:
        """Add metrics entries from distributed fit, indexed by client and timestamp.

        Args:
            client_id: The client identifier
            metrics: Dictionary of metric name to value
            timestamp: The timestamp when these metrics were recorded
        """
        for key in metrics:
            if key not in self.metrics_distributed_fit_async:
                self.metrics_distributed_fit_async[key] = {}
            if client_id not in self.metrics_distributed_fit_async[key]:
                self.metrics_distributed_fit_async[key][client_id] = []
            self.metrics_distributed_fit_async[key][client_id].append(
                (timestamp, metrics[key])
            )

    def add_metrics_centralized_async(
        self, metrics: Dict[str, Scalar], timestamp: float
    ) -> None:
        """Add metrics entries from centralized evaluation with timestamp.

        Args:
            metrics: Dictionary of metric name to value
            timestamp: The timestamp when these metrics were recorded
        """
        for metric in metrics:
            if metric not in self.metrics_centralized_async:
                self.metrics_centralized_async[metric] = []
            self.metrics_centralized_async[metric].append((timestamp, metrics[metric]))

    def add_loss_centralized_async(self, timestamp: float, loss: float) -> None:
        """Add loss entry from centralized evaluation with timestamp.

        Args:
            timestamp: The timestamp when this loss was recorded
            loss: The loss value
        """
        self.losses_centralized_async.append((timestamp, loss))

    # Override parent methods to maintain compatibility with flwr 1.25.0+ API
    # The parent class now uses server_round: int as first parameter

    def add_loss_centralized(self, server_round: int, loss: float) -> None:
        """Add one loss entry from centralized evaluation.

        Args:
            server_round: The server round number (or can be used as timestamp for async)
            loss: The loss value
        """
        super().add_loss_centralized(server_round=server_round, loss=loss)

    def add_loss_distributed(self, server_round: int, loss: float) -> None:
        """Add one loss entry from distributed evaluation.

        Args:
            server_round: The server round number (or can be used as timestamp for async)
            loss: The loss value
        """
        super().add_loss_distributed(server_round=server_round, loss=loss)

    def add_metrics_centralized(
        self, server_round: int, metrics: Dict[str, Scalar]
    ) -> None:
        """Add metrics entries from centralized evaluation.

        Args:
            server_round: The server round number (or can be used as timestamp for async)
            metrics: Dictionary of metric name to value
        """
        super().add_metrics_centralized(server_round=server_round, metrics=metrics)

    def add_metrics_distributed(
        self, server_round: int, metrics: Dict[str, Scalar]
    ) -> None:
        """Add metrics entries from distributed evaluation.

        Args:
            server_round: The server round number (or can be used as timestamp for async)
            metrics: Dictionary of metric name to value
        """
        super().add_metrics_distributed(server_round=server_round, metrics=metrics)
