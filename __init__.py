"""Flower Async - Asynchronous Federated Learning with Flower.

This package provides asynchronous federated learning components
compatible with the Flower (flwr) framework.

Components:
    - AsyncServer: Asynchronous FL server
    - AsyncClientManager: Client manager with free/busy state tracking
    - AsynchronousStrategy: Async aggregation strategies (FedAsync, etc.)
    - AsyncHistory: Extended history tracking with timestamps
    - AsyncVisualizer: Visualization utilities for async FL (requires monitoring_sync)
"""

try:
    # When used as an installed package
    from .async_server import AsyncServer
    from .async_client_manager import AsyncClientManager
    from .async_strategy import AsynchronousStrategy
    from .async_history import AsyncHistory
except ImportError:
    # When imported directly from the directory
    from async_server import AsyncServer
    from async_client_manager import AsyncClientManager
    from async_strategy import AsynchronousStrategy
    from async_history import AsyncHistory

# AsyncVisualizer has external dependency on monitoring_sync
try:
    try:
        from .async_visualizer import AsyncVisualizer
    except ImportError:
        from async_visualizer import AsyncVisualizer
except ImportError:
    AsyncVisualizer = None  # type: ignore[misc, assignment]

__all__ = [
    "AsyncServer",
    "AsyncClientManager",
    "AsynchronousStrategy",
    "AsyncHistory",
    "AsyncVisualizer",
]

__version__ = "0.1.0"
