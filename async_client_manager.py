"""Async Client Manager for Flower asynchronous federated learning.

This module provides an AsyncClientManager that extends SimpleClientManager
to support tracking of free/busy client states needed for asynchronous FL.
"""

from typing import Dict, List, Optional

import random
import threading
from logging import INFO, ERROR

from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion


class AsyncClientManager(SimpleClientManager):
    """Client manager for asynchronous federated learning.

    This class extends SimpleClientManager to track which clients are currently
    free (available for training) vs busy (currently training). This is essential
    for asynchronous FL where clients can start and finish at different times.
    """

    def __init__(self) -> None:
        """Initialize the AsyncClientManager."""
        super().__init__()
        self.free_clients: Dict[str, ClientProxy] = {}
        self._cv_free = threading.Condition()

    def set_client_to_busy(self, client_id: str) -> bool:
        """Mark a client as busy (currently training).

        Args:
            client_id: The client identifier to mark as busy

        Returns:
            True if successful, False if client not found
        """
        if client_id not in self.free_clients or client_id not in self.clients:
            log(ERROR, "Client not found in free_clients")
            return False
        else:
            with self._cv_free:
                self.free_clients.pop(client_id)
                self._cv_free.notify_all()
            return True

    def set_client_to_free(self, client_id: str) -> bool:
        """Mark a client as free (available for training).

        Args:
            client_id: The client identifier to mark as free

        Returns:
            True if successful, False if client not found
        """
        if client_id not in self.clients:
            log(ERROR, "Client not found in clients")
            return False
        else:
            with self._cv_free:
                self.free_clients[client_id] = self.clients[client_id]
                self._cv_free.notify_all()
            return True

    def wait_for_free(self, num_free_clients: int, timeout: int = 86400) -> bool:
        """Wait until the specified number of clients are free.

        Args:
            num_free_clients: Number of free clients to wait for
            timeout: Maximum time to wait in seconds (default: 86400 = 24h)

        Returns:
            True if condition was met, False if timeout occurred
        """
        with self._cv_free:
            return self._cv_free.wait_for(
                lambda: len(self.free_clients) >= num_free_clients, timeout=5
            )

    def register(self, client: ClientProxy) -> bool:
        """Register a Flower ClientProxy instance.

        Args:
            client: The ClientProxy to register

        Returns:
            True if registration was successful, False otherwise
        """
        log(INFO, "Registering client with id: %s", client.cid)
        if super().register(client):
            self.set_client_to_free(client.cid)
            return True
        else:
            return False

    def unregister(self, client: ClientProxy) -> None:
        """Unregister a Flower ClientProxy instance.

        Args:
            client: The ClientProxy to unregister
        """
        log(INFO, "Unregistering client with id: %s", client.cid)
        if client.cid in self.free_clients:
            self.set_client_to_busy(client.cid)
        super().unregister(client)

    def num_free(self) -> int:
        """Return the number of free clients.

        Returns:
            Number of currently free clients
        """
        return len(self.free_clients)

    def all_free(self) -> List[str]:
        """Return list of all free client IDs.

        Returns:
            List of free client identifiers
        """
        return list(self.free_clients.keys())

    def sample_free(
        self,
        num_free_clients: int,
        min_num_free_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of free Flower ClientProxy instances.

        Args:
            num_free_clients: Number of clients to sample
            min_num_free_clients: Minimum number of free clients required
            criterion: Optional criterion for filtering clients

        Returns:
            List of sampled ClientProxy instances
        """
        log(INFO, "Sampling %s clients, min %s", num_free_clients, min_num_free_clients)
        # Block until at least num_clients are free. It always samples from all clients.
        if min_num_free_clients is None:
            min_num_free_clients = num_free_clients
        self.wait_for_free(min_num_free_clients)

        # Sample clients which meet the criterion
        available_cids = list(self.free_clients.keys())
        if criterion is not None:
            available_cids = [
                cid
                for cid in available_cids
                if criterion.select(self.free_clients[cid])
            ]

        if num_free_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available free clients"
                " (%s) is less than number of requested free clients (%s).",
                len(available_cids),
                num_free_clients,
            )
            return []

        sampled_cids = random.sample(available_cids, num_free_clients)
        ret_list = [self.free_clients[cid] for cid in sampled_cids]
        for cid in sampled_cids:
            self.set_client_to_busy(cid)
        return ret_list

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances.

        This method delegates to sample_free to sample from available clients.

        Args:
            num_clients: Number of clients to sample
            min_num_clients: Minimum number of clients required
            criterion: Optional criterion for filtering clients

        Returns:
            List of sampled ClientProxy instances
        """
        return self.sample_free(num_clients, min_num_clients, criterion)
