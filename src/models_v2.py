"""Spiking Neural Network (SNN) model."""

import math
import random
from typing import List, Optional

import networkx as nx
import numpy as np


class SNN:
    """Spiking Neural Network (SNN) model."""

    def __init__(
        self,
        num_neurons: int,
        membrane_threshold: float,
        current_period: float,
        external_current: float,
        weights_mean: float,
        simulation_duration: int,
        refractory_period: float,
        leak_coefficient: float,
        small_world_graph_p: float = 0.1,
        small_world_graph_k: int = 10,
    ) -> None:
        """Initialize the spiking neural network (SNN) with parameters."""
        self.tot_spikes: int = 0
        self.leak_refractory_ratio: float = leak_coefficient / refractory_period
        self.num_neurons: int = num_neurons
        self.membrane_threshold: float = membrane_threshold
        self.current_period_times_refractory: float = current_period * refractory_period
        self.external_current: float = external_current
        self.simulation_duration: int = simulation_duration
        self.refractory_period: float = refractory_period
        self.time_step: int = 1
        self.weights_mean: float = weights_mean

        self.membrane_potentials: np.ndarray = np.random.uniform(
            0, self.membrane_threshold, self.num_neurons
        )
        self.spike_times: List[List[int]] = [[] for _ in range(num_neurons)]
        self.avg_in_degree: Optional[float] = None
        self.spike_matrix: Optional[np.ndarray] = None

        self.generate_synaptic_weights(small_world_graph_p, small_world_graph_k)

        self.refractory_timer: np.ndarray = np.zeros(num_neurons)

    def generate_synaptic_weights(
        self,
        small_world_graph_p: float = 0.1,
        small_world_graph_k: int = 10,
        weights_scale_factor: float = 0.1,
    ) -> None:
        """Generate synaptic weights based on a small-world graph."""
        small_world_graph = nx.watts_strogatz_graph(
            n=self.num_neurons, k=small_world_graph_k, p=small_world_graph_p, seed=None
        )
        synaptic_weights = np.zeros((self.num_neurons, self.num_neurons))

        for edge in small_world_graph.edges():
            i, j = edge
            if np.random.rand() < 0.5:
                synaptic_weights[i, j] = np.random.normal(
                    loc=self.weights_mean,
                    scale=abs(self.weights_mean) / weights_scale_factor,
                )
            else:
                synaptic_weights[j, i] = np.random.normal(
                    loc=self.weights_mean,
                    scale=abs(self.weights_mean) / weights_scale_factor,
                )

        np.fill_diagonal(synaptic_weights, 0)
        self.synaptic_weights: np.ndarray = synaptic_weights

        in_degrees = np.count_nonzero(self.synaptic_weights, axis=0)
        self.avg_in_degree = in_degrees.mean()

    def in_degree(self) -> Optional[float]:
        """Return the average in-degree of the network."""
        return self.avg_in_degree

    def stimulate_neuron(self, num_stimulated_neurons: int) -> None:
        """Stimulate a specified number of neurons."""
        target_neurons = random.sample(range(self.num_neurons), num_stimulated_neurons)
        self.membrane_potentials[target_neurons] += self.external_current

    def simulate(self) -> Optional[np.ndarray]:
        """Run the simulation of the SNN."""
        self.tot_spikes = 0
        self.spike_matrix = np.zeros(
            (self.simulation_duration, self.num_neurons), dtype=int
        )
        integer_current_period_times_refractory = int(
            self.current_period_times_refractory
        )
        greatest_common_divisor = math.gcd(
            int(self.current_period_times_refractory * 10), 10
        )
        tau_n = int(self.current_period_times_refractory * 10 / greatest_common_divisor)
        tau_d = int(10 / greatest_common_divisor)
        currents_counter = 0

        for t in range(self.simulation_duration):
            self.refractory_timer = np.maximum(
                0, self.refractory_timer - self.time_step
            )

            if self.current_period_times_refractory < 1:
                if currents_counter % tau_n == 0:
                    self.stimulate_neuron(tau_d)
            if self.current_period_times_refractory >= 1:
                if (
                    integer_current_period_times_refractory > 0
                    and currents_counter % integer_current_period_times_refractory == 0
                ):
                    self.stimulate_neuron(1)
                if currents_counter == self.current_period_times_refractory * 10:
                    currents_counter = 0
            currents_counter += 1

            spiking_neurons = (self.membrane_potentials >= self.membrane_threshold) & (
                self.refractory_timer == 0
            )
            self.spike_matrix[t, :] = spiking_neurons.astype(int)
            self.tot_spikes += np.sum(spiking_neurons)

            for idx in np.where(spiking_neurons)[0]:
                self.spike_times[idx].append(t)

            self.membrane_potentials[spiking_neurons] = 0
            self.refractory_timer[spiking_neurons] = self.refractory_period + 1
            self.membrane_potentials = (
                1 - self.leak_refractory_ratio
            ) * self.membrane_potentials + spiking_neurons.astype(
                float
            ) @ self.synaptic_weights
        return self.spike_matrix

    def calculate_mean_isi(self) -> float:
        """Calculate the mean inter-spike interval (ISI)."""
        total_inter_spike_intervals = []
        for spike_times in self.spike_times:
            if len(spike_times) > 1:
                inter_spike_interval = np.diff(spike_times)
                total_inter_spike_intervals.extend(inter_spike_interval)
        if total_inter_spike_intervals:
            mean_inter_spike_interval = np.mean(total_inter_spike_intervals)
        else:
            mean_inter_spike_interval = 0
        return float(mean_inter_spike_interval) / self.refractory_period


def calculate_weights_mean(
    delta_mean: List[float],
    leak_coefficient: float,
    membrane_threshold: float,
    external_current: float,
    current_period: float,
    num_neurons: int,
    refractory_period: float,
    beta: float,
) -> np.ndarray:
    """Calculate weights mean based on parameters."""
    epsilon = 1e-10
    delta_mean = np.array(delta_mean, dtype=np.float64)
    denominator_exp = 1 - np.exp(-leak_coefficient * delta_mean)
    denominator_exp = np.where(denominator_exp < epsilon, epsilon, denominator_exp)

    numerator = (
        leak_coefficient * membrane_threshold * delta_mean**2
    ) / denominator_exp - (external_current * delta_mean**2) / (
        current_period * num_neurons
    )

    denominator = beta * (delta_mean - refractory_period)
    denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)

    return numerator / denominator
