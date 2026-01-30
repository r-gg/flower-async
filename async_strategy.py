"""Asynchronous aggregation strategies for Flower FL.

This module provides asynchronous aggregation strategies for federated learning,
implementing FedAsync and related algorithms.

References:
    - FedAsync: https://arxiv.org/pdf/1903.03934.pdf
    - PAFLM: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9022982
    - AsyncFedED: https://arxiv.org/pdf/2205.13797.pdf
"""

import math
from typing import List, Tuple
from time import time
from logging import DEBUG, WARNING

from flwr.common import (
    Parameters,
    NDArrays,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    log,
)
from flwr.server.strategy.aggregate import aggregate


class AsynchronousStrategy:
    """Asynchronous aggregation strategy for federated learning.

    This class implements various asynchronous aggregation methods including
    FedAsync with staleness weighting and sample weighting.

    Attributes:
        total_samples: Total number of samples across all clients
        staleness_alpha: Staleness decay parameter
        fedasync_mixing_alpha: Mixing coefficient for FedAsync
        fedasync_a: Polynomial decay parameter
        num_clients: Number of clients
        async_aggregation_strategy: The aggregation strategy to use
        use_staleness: Whether to use staleness-based weighting
        use_sample_weighing: Whether to use sample-based weighting
        send_gradients: Whether clients send gradients instead of models
        server_artificial_delay: Whether to add artificial delay for testing
    """

    def __init__(
        self,
        total_samples: int,
        staleness_alpha: float,
        fedasync_mixing_alpha: float,
        fedasync_a: float,
        num_clients: int,
        async_aggregation_strategy: str,
        use_staleness: bool,
        use_sample_weighing: bool,
        send_gradients: bool,
        server_artificial_delay: bool,
    ) -> None:
        """Initialize the AsynchronousStrategy.

        Args:
            total_samples: Total samples across all clients
            staleness_alpha: Staleness decay parameter
            fedasync_mixing_alpha: Mixing coefficient for FedAsync
            fedasync_a: Polynomial decay parameter
            num_clients: Number of clients
            async_aggregation_strategy: Strategy name ('fedasync', etc.)
            use_staleness: Enable staleness-based weighting
            use_sample_weighing: Enable sample-based weighting
            send_gradients: True if clients send gradients, False for models
            server_artificial_delay: Add artificial delay for testing
        """
        self.total_samples = total_samples
        self.staleness_alpha = staleness_alpha
        self.fedasync_a = fedasync_a
        self.fedasync_mixing_alpha = fedasync_mixing_alpha
        self.num_clients = num_clients
        self.async_aggregation_strategy = async_aggregation_strategy
        self.use_staleness = use_staleness
        self.use_sample_weighing = use_sample_weighing
        self.send_gradients = send_gradients
        self.server_artificial_delay = server_artificial_delay

    def average(
        self,
        global_parameters: Parameters,
        model_update_parameters: Parameters,
        t_diff: float,
        num_samples: int,
    ) -> Parameters:
        """Compute the average of the global and client parameters.

        Args:
            global_parameters: Current global model parameters
            model_update_parameters: Parameters from client update
            t_diff: Time difference (staleness)
            num_samples: Number of samples used by client

        Returns:
            Aggregated parameters
        """
        if self.async_aggregation_strategy == "fedasync":
            if self.send_gradients:
                return self.weighted_merge_fedasync(
                    global_parameters, model_update_parameters, t_diff, num_samples
                )
            else:
                return self.weighted_average_fedasync(
                    global_parameters, model_update_parameters, t_diff, num_samples
                )
        else:
            raise ValueError(
                f"Invalid async aggregation strategy: {self.async_aggregation_strategy}"
            )

    def unweighted_average(
        self, global_parameters: Parameters, model_update_parameters: Parameters
    ) -> Parameters:
        """Compute the unweighted average of the global and client parameters.

        Args:
            global_parameters: Current global model parameters
            model_update_parameters: Parameters from client update

        Returns:
            Averaged parameters
        """
        return ndarrays_to_parameters(
            aggregate(
                [
                    (parameters_to_ndarrays(global_parameters), 1),
                    (parameters_to_ndarrays(model_update_parameters), 1),
                ]
            )
        )

    def weighted_average_asyncfeded(
        self,
        global_parameters: Parameters,
        model_update_parameters: Parameters,
        t_diff: float,
        num_samples: int,
    ) -> Parameters:
        """Compute weighted average inspired by AsyncFedED.

        Reference: https://arxiv.org/pdf/2205.13797.pdf

        Args:
            global_parameters: Current global model parameters
            model_update_parameters: Parameters from client update
            t_diff: Time difference (staleness)
            num_samples: Number of samples used by client

        Returns:
            Aggregated parameters
        """
        return ndarrays_to_parameters(
            self.aggregate_asyncfeded(
                parameters_to_ndarrays(global_parameters),
                parameters_to_ndarrays(model_update_parameters),
                t_diff,
                num_samples=num_samples,
            )
        )

    def get_sample_weight_coeff(self, num_samples: int) -> float:
        """Compute the sample weight coefficient.

        Args:
            num_samples: Number of samples used by client

        Returns:
            Sample weight coefficient
        """
        return num_samples / self.total_samples

    def weighted_average_fedasync(
        self,
        global_parameters: Parameters,
        model_update_parameters: Parameters,
        t_diff: float,
        num_samples: int,
    ) -> Parameters:
        """Compute weighted average inspired by FedAsync.

        Reference: https://arxiv.org/pdf/1903.03934.pdf

        Args:
            global_parameters: Current global model parameters
            model_update_parameters: Parameters from client update
            t_diff: Time difference (staleness)
            num_samples: Number of samples used by client

        Returns:
            Aggregated parameters
        """
        return ndarrays_to_parameters(
            self.aggregate_fedasync(
                parameters_to_ndarrays(global_parameters),
                parameters_to_ndarrays(model_update_parameters),
                t_diff,
                num_samples=num_samples,
            )
        )

    def busy_wait(self, seconds: float) -> None:
        """Busy wait for the specified number of seconds.

        Args:
            seconds: Number of seconds to wait
        """
        start = time()
        while time() - start < seconds:
            pass

    def aggregate_fedasync(
        self,
        global_param_arr: NDArrays,
        model_update_param_arr: NDArrays,
        t_diff: float,
        num_samples: int,
    ) -> NDArrays:
        """Compute weighted average with FedAsync formula.

        Formula: params_new = (1-alpha) * params_old + alpha * model_update_params

        Args:
            global_param_arr: Global model parameters as NDArrays
            model_update_param_arr: Client model parameters as NDArrays
            t_diff: Time difference (staleness)
            num_samples: Number of samples used by client

        Returns:
            Aggregated parameters as NDArrays
        """
        alpha_coeff = self.fedasync_mixing_alpha
        if self.use_staleness:
            alpha_coeff *= self.get_staleness_weight_coeff_fedasync_poly(t_diff=t_diff)
        if self.use_sample_weighing:
            alpha_coeff *= self.get_sample_weight_coeff(num_samples)

        if self.server_artificial_delay:
            self.busy_wait(0.5)

        return [
            (1 - alpha_coeff) * layer_global + alpha_coeff * layer_update
            for layer_global, layer_update in zip(
                global_param_arr, model_update_param_arr
            )
        ]

    def weighted_merge_fedasync(
        self,
        global_parameters: Parameters,
        gradients: Parameters,
        t_diff: float,
        num_samples: int,
    ) -> Parameters:
        """Add gradients to the global model with FedAsync weighting.

        Note: This differs from original FedAsync which aggregates models;
        here we aggregate gradients instead.

        Reference: https://arxiv.org/pdf/1903.03934.pdf

        Args:
            global_parameters: Current global model parameters
            gradients: Gradients from client
            t_diff: Time difference (staleness)
            num_samples: Number of samples used by client

        Returns:
            Updated parameters
        """
        if self.server_artificial_delay:
            self.busy_wait(1)
        return ndarrays_to_parameters(
            self.add_grads_fedasync(
                parameters_to_ndarrays(global_parameters),
                parameters_to_ndarrays(gradients),
                t_diff,
                num_samples=num_samples,
            )
        )

    def add_grads_fedasync(
        self,
        global_param_arr: NDArrays,
        gradients_arr: NDArrays,
        t_diff: float,
        num_samples: int,
    ) -> NDArrays:
        """Compute weighted update with gradients.

        Formula: params_new = (1-alpha) * params_old + alpha * (params_old + grads)

        Args:
            global_param_arr: Global model parameters as NDArrays
            gradients_arr: Gradients as NDArrays
            t_diff: Time difference (staleness)
            num_samples: Number of samples used by client

        Returns:
            Updated parameters as NDArrays
        """
        alpha_coeff = self.fedasync_mixing_alpha
        if self.use_staleness:
            alpha_coeff *= self.get_staleness_weight_coeff_fedasync_poly(t_diff=t_diff)
        if self.use_sample_weighing:
            alpha_coeff *= self.get_sample_weight_coeff(num_samples)

        return [
            (1 - alpha_coeff) * layer_global + alpha_coeff * (layer_global + layer_grad)
            for layer_global, layer_grad in zip(global_param_arr, gradients_arr)
        ]

    def aggregate_asyncfeded(
        self,
        global_param_arr: NDArrays,
        model_update_param_arr: NDArrays,
        t_diff: float,
        num_samples: int,
    ) -> NDArrays:
        """Compute parameters using AsyncFedED formula.

        Formula: params_new = params_old + nu * (model_update_params - params_old)
        where nu is influenced by staleness and/or sample counts.

        Reference: https://arxiv.org/pdf/2205.13797.pdf

        Args:
            global_param_arr: Global model parameters as NDArrays
            model_update_param_arr: Client model parameters as NDArrays
            t_diff: Time difference (staleness)
            num_samples: Number of samples used by client

        Returns:
            Updated parameters as NDArrays
        """
        eta = 1.0
        if self.use_staleness:
            eta *= self.get_staleness_weight_coeff_paflm(t_diff=t_diff)
        if self.use_sample_weighing:
            eta *= self.get_sample_weight_coeff(num_samples)

        log(DEBUG, f"t_diff: {t_diff}\nnu: {eta}")
        return [
            layer_global + eta * (layer_update - layer_global)
            for layer_global, layer_update in zip(
                global_param_arr, model_update_param_arr
            )
        ]

    def get_staleness_weight_coeff_paflm(self, t_diff: float) -> float:
        """Compute staleness weight using PAFLM formula.

        Reference: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9022982

        Args:
            t_diff: Time difference (staleness)

        Returns:
            Staleness weight coefficient
        """
        mu_staleness = t_diff
        exponent = ((1 / float(self.num_clients)) * mu_staleness) - 1
        beta_P = math.pow(self.staleness_alpha, exponent)
        return beta_P

    def get_staleness_weight_coeff_fedasync_constant(self) -> float:
        """Return constant staleness weight (1.0).

        Reference: https://arxiv.org/pdf/1903.03934.pdf

        Returns:
            Constant weight of 1.0
        """
        return 1.0

    def get_staleness_weight_coeff_fedasync_poly(self, t_diff: float) -> float:
        """Compute staleness weight using polynomial decay.

        Formula: (t_diff + 1)^(-a)

        Reference: https://arxiv.org/pdf/1903.03934.pdf

        Args:
            t_diff: Time difference (staleness)

        Returns:
            Staleness weight coefficient
        """
        return math.pow(t_diff + 1, -self.fedasync_a)

    def get_staleness_weight_coeff_fedasync_hinge(
        self, t_diff: float, a: float = 10, b: float = 4
    ) -> float:
        """Compute staleness weight using hinge function.

        Formula: 1 if t_diff <= b else 1 / (a * (t_diff - b) + 1)

        Reference: https://arxiv.org/pdf/1903.03934.pdf

        Args:
            t_diff: Time difference (staleness)
            a: Decay rate parameter
            b: Threshold parameter

        Returns:
            Staleness weight coefficient
        """
        return 1 if t_diff <= b else 1 / (a * (t_diff - b) + 1)
