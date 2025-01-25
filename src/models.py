import numpy as np
import random
from dataclasses import dataclass, field
from typing import List

@dataclass
class SNNParameters:
    """
    Parameters for the Spiking Neural Network (SNN).
    """
    num_neurons: int = 1000          # Number of neurons
    theta: float = 7.0               # Firing threshold
    tau: float = 10.0                # Time interval for external current
    external_current: float = 0.5    # Intensity of external current
    t_ref: float = 2.0               # Refractory period
    leak: float = 0.0                # Leak parameter (for leaky integrate-and-fire)
    w_mean: float = 0.0              # Initial mean synaptic weight (updated if needed)
    w_std_coefficient: float = 1.0               # Std dev for the synaptic weights
    simulation_steps: int = 1000     # Number of steps in the simulation

@dataclass
class SNN:
    """
    Spiking Neural Network (SNN) model.
    """
    params: SNNParameters
    membrane_potentials: np.ndarray = field(init=False)
    synaptic_weights: np.ndarray = field(init=False)
    spike_times: List[List[int]] = field(init=False)  # Each neuron has a list of spike times
    refractory_timer: np.ndarray = field(init=False)
    tot_spike: int = 0

    def __post_init__(self) -> None:
        self.membrane_potentials = np.random.uniform(
            low=0, high=self.params.theta, size=self.params.num_neurons
        )
        self.synaptic_weights = np.random.normal(
            loc=self.params.w_mean,
            scale=self.params.w_std_coefficient * self.params.w_mean,
            size=(self.params.num_neurons, self.params.num_neurons)
        )
        
        # Prepare a list of lists for spike times (one list per neuron)
        self.spike_times = [[] for _ in range(self.params.num_neurons)]
        
        # No self-connection
        np.fill_diagonal(self.synaptic_weights, 0)
        
        # Refractory timers start at 0
        self.refractory_timer = np.zeros(self.params.num_neurons)

    def stimulate_neuron(self) -> None:
        """Deliver external current to a randomly chosen neuron."""
        target_neuron = random.randint(0, self.params.num_neurons - 1)
        self.membrane_potentials[target_neuron] += self.params.external_current

    def simulate(self) -> None:
        """
        Simulate the network's activity over `simulation_steps`.
        Each iteration:
          - Decrease refractory_timer by 1
          - Possibly stimulate a random neuron
          - Check for spiking neurons
          - Update membrane potentials and refractory periods
        """
        self.spike_matrix = np.zeros(
            (self.params.simulation_steps, self.params.num_neurons), dtype=int
        )
        self.tot_spike = 0

        for t in range(self.params.simulation_steps):
            # Decrement refractory timers
            self.refractory_timer = np.maximum(0, self.refractory_timer - 1)

            # Stimulate a neuron at specific intervals
            if t % self.params.tau == 0:
                self.stimulate_neuron()

            # Determine which neurons spike
            spiking_neurons = (
                (self.membrane_potentials >= self.params.theta)
                & (self.refractory_timer == 0)
            )

            # Record spikes in the spike matrix
            self.spike_matrix[t, :] = spiking_neurons.astype(int)
            self.tot_spike += np.sum(spiking_neurons)

            # Append spike times for spiking neurons
            for idx in np.where(spiking_neurons)[0]:
                self.spike_times[idx].append(t)

            # Reset potentials for spiking neurons & set refractory timers
            self.membrane_potentials[spiking_neurons] = 0
            self.refractory_timer[spiking_neurons] = self.params.t_ref

            # Update membrane potentials:
            #   - Add synaptic input from spiking neurons
            #   - Subtract leak
            self.membrane_potentials += (
                spiking_neurons.astype(float) @ self.synaptic_weights
                - self.params.leak * self.membrane_potentials
            )

    def get_mean_isi(self) -> float:
        """
        Return the mean inter-spike interval (ISI) across all neurons in the network.
        If no spikes or single spikes per neuron, returns 0.
        """
        total_inter_spike_intervals = []
        for neuron_spikes in self.spike_times:
            if len(neuron_spikes) > 1:
                isi = np.diff(neuron_spikes)
                total_inter_spike_intervals.extend(isi)

        if total_inter_spike_intervals:
            return float(np.mean(total_inter_spike_intervals))
        else:
            return 0.0

    def get_total_spikes(self) -> int:
        """Return the total number of spikes over the entire simulation."""
        return self.tot_spike

def lzw_complexity_from_matrix(matrix: np.ndarray) -> int:
    """
    Calculate the Lempel-Ziv-Welch (LZW) complexity of a vector created by
    concatenating the columns of a 2D matrix.
    Args:
        matrix (np.ndarray): A 2D NumPy array representing spike data
                             (rows typically time, columns neurons).

    Returns:
        int: The LZW complexity of the concatenated sequence.
    """
    def lzw(seq: str) -> int:
        """
        Calculate the LZW complexity of a binary (string) sequence.

        Args:
            seq (str): The sequence string (e.g., '101001...').

        Returns:
            int: The size of the generated dictionary, representing
                 the LZW complexity.
        """
        dictionary = {}
        w = ""
        for c in seq:
            wc = w + c
            if wc not in dictionary:
                dictionary[wc] = len(dictionary)
                w = c
            else:
                w = wc
        return len(dictionary)

    # Transpose and then flatten to read column by column
    vector = matrix.T.flatten()
    # Convert the vector into a string
    vector_str = "".join(map(str, vector))

    # Calculate the LZW complexity
    complexity = lzw(vector_str)
    return complexity
