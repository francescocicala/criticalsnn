"""Spiking Neural Network (SNN) Self Organized quasi-Criticality (SOqC)."""

import random
import numpy as np
import logging
import sys
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class SoqcParams:
    """Simulation parameters for the Spiking Neural Network (SNN)."""

    num_neurons: int
    membrane_threshold: float
    currents_period: int
    external_current: float
    number_of_external_current: int
    leak_coefficient: float
    simulation_duration: int
    refractory_period: float
    guessed_critical_weight: float
    precision: float

    def __post_init__(self):
        """Validate conditions based on parameter constraints."""

        logging.info("Validating conditions with parameter")

        if (self.number_of_external_current * self.currents_period *
            self.num_neurons * self.membrane_threshold *
            self.leak_coefficient == 0):
            logging.warning("One or more parameters are equal to zero.")
            sys.exit()
            return False

        if (self.external_current /
            ((self.currents_period / self.number_of_external_current) *
             self.num_neurons * self.leak_coefficient * self.membrane_threshold) < 1):
            logging.warning(
                "Condition violated: I / (tau * num_neurons * leak_coefficient * theta) < 1"
            )
            sys.exit()
            return False

        if (2 * self.external_current /
            ((self.currents_period / self.number_of_external_current) *
             self.num_neurons * self.membrane_threshold) > 1):
            logging.warning("Condition violated: 2 * I / (tau * num_neurons * theta) > 1")
            sys.exit()
            return False

        if 1 / (2 * self.leak_coefficient) < 1:
            logging.warning("Condition violated: 1 / (2 * leak_coefficient) < 1")
            sys.exit()
            return False

        logging.info("All conditions validated successfully.")
        return True


class SNN:
    """Spiking Neural Network (SNN) model."""

    def __init__(self, simulation_params: SoqcParams, save_history=False) -> None:
        """Initialize the spiking neural network (SNN) with parameters."""

        self.tot_spikes: int = 0
        self.leak_refractory_ratio: float = (
            simulation_params.leak_coefficient / simulation_params.refractory_period
        )
        self.num_neurons: int = simulation_params.num_neurons
        self.membrane_threshold: float = simulation_params.membrane_threshold
        self.current_period_times_refractory: float = (
            simulation_params.currents_period * simulation_params.refractory_period
        )
        self.external_current: float = simulation_params.external_current
        self.simulation_duration: int = simulation_params.simulation_duration
        self.refractory_period: float = simulation_params.refractory_period
        self.time_step: int = 1
        self.membrane_potentials: np.ndarray = np.random.uniform(
            0, self.membrane_threshold, self.num_neurons
        )
        self.spike_times: List[List[int]] = [
            [] for _ in range(simulation_params.num_neurons)
        ]
        self.spike_matrix: Optional[np.ndarray] = None
        self.refractory_timer: np.ndarray = np.zeros(simulation_params.num_neurons)
        self.avg_in_degree: int = simulation_params.num_neurons - 1
        self.guessed_critical_weight: float = simulation_params.guessed_critical_weight
        self.number_of_external_current: int = simulation_params.number_of_external_current
        self.currents_period: int = simulation_params.currents_period
        self.percentage_weights_update = simulation_params.precision
        self.save_history = save_history

        self.WEIGHTS_SCALE_FACTOR: float = 0.1
        self.PERCENTAGE_FIRING_NEURONS_AT_CRITICALITY = 0.5
        self.TO_BE_IGNORED_FOR_HISTORY = 0.5

    def theoretical_critical_weight(self):
        """Returns the synaptic weight corresponding to the critical point."""
        return (
            self.membrane_threshold / self.avg_in_degree
            - 2 * self.external_current * self.refractory_period
            / (self.currents_period * self.num_neurons * self.avg_in_degree)
        )

    def in_degree(self) -> Optional[float]:
        """Return the average in-degree of the network."""
        return self.avg_in_degree

    def stimulate_neurons(self) -> None:
        """Stimulate a specified number of neurons."""
        target_neurons = random.sample(range(self.num_neurons), self.number_of_external_current)
        self.membrane_potentials[target_neurons] += self.external_current

    def simulate(self) -> Optional[np.ndarray]:
        """Run the simulation of the SNN."""
        self.tot_spikes = 0
        self.spike_matrix = np.zeros(
            (self.simulation_duration, self.num_neurons), dtype=int
        )
        computed_mean_weight = self.guessed_critical_weight
        if self.save_history:
            weights_history = []

        for t in range(self.simulation_duration):
            synaptic_weights = np.random.normal(
                loc=computed_mean_weight,
                scale=computed_mean_weight * self.WEIGHTS_SCALE_FACTOR,
                size=(self.num_neurons, self.num_neurons)
            )
            np.fill_diagonal(synaptic_weights, 0)

            self.refractory_timer = np.maximum(0, self.refractory_timer - self.time_step)

            if t % self.currents_period == 0:
                self.stimulate_neurons()

            spiking_neurons = (self.membrane_potentials >= self.membrane_threshold) & (
                self.refractory_timer == 0
            )
            self.tot_spikes += np.sum(spiking_neurons)

            if t % (self.refractory_period + 1) == 0:
                computed_mean_weight -= (computed_mean_weight * self.percentage_weights_update
                                          * (self.tot_spikes - self.num_neurons * self.PERCENTAGE_FIRING_NEURONS_AT_CRITICALITY)
                                          / (self.num_neurons * self.PERCENTAGE_FIRING_NEURONS_AT_CRITICALITY)
                                          )
                self.tot_spikes = 0

            for idx in np.where(spiking_neurons)[0]:
                self.spike_times[idx].append(t)

            self.membrane_potentials[spiking_neurons] = 0
            self.refractory_timer[spiking_neurons] = self.refractory_period + 1
            self.membrane_potentials = (
                1 - self.leak_refractory_ratio
            ) * self.membrane_potentials + spiking_neurons.astype(float) @ synaptic_weights

            if self.save_history and t > self.simulation_duration * self.TO_BE_IGNORED_FOR_HISTORY:
                weights_history.append(computed_mean_weight)

        if self.save_history:
            return weights_history
        else:
            return computed_mean_weight
