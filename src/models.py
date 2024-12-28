import numpy as np
import random
from dataclasses import dataclass, field

@dataclass
class SNNParameters:
    """
    Parameters for the Spiking Neural Network (SNN).
    """
    num_neurons: int = 1000  # Number of neurons
    theta: float = 7.0  # Firing threshold
    tau: float = 10.0  # Time interval for external current
    external_current: float = 0.5  # Intensity of external current
    t_ref: float = 2.0  # Refractory period
    leak: float = 0.  # Leak parameter for leaky integrate-and-fire
    w_mean: float = 0.0  # Initial mean synaptic weight (will be updated)
    w_std: float = 1 # std dev parameter for the synaptic weights
    simulation_steps: int = 1000

@dataclass
class SNN:
    """
    Spiking Neural Network (SNN) model.
    """
    params: SNNParameters
    membrane_potentials: np.ndarray = field(init=False)
    synaptic_weights: np.ndarray = field(init=False)
    spike_times: list = field(init=False)
    refractory_timer: np.ndarray = field(init=False)
    tot_spike: int = 0

    def __post_init__(self):
        self.membrane_potentials = np.random.uniform(0, self.params.theta, self.params.num_neurons)
        self.synaptic_weights = np.random.normal(loc=self.params.w_mean, scale=self.params.w_std, size=(self.params.num_neurons, self.params.num_neurons))
        
        # TODO: initialize it to a (num_neurons, simulation_steps)-array for better parallelization.
        self.spike_times = [[] for _ in range(self.params.num_neurons)]
        np.fill_diagonal(self.synaptic_weights, 0)
        self.refractory_timer = np.zeros(self.params.num_neurons)

    def stimulate_neuron(self):
        """Deliver external current to a random neuron."""
        target_neuron = random.randint(0, self.params.num_neurons - 1)
        self.membrane_potentials[target_neuron] += self.params.external_current

    def simulate(self):
        """Simulate the network's activity over the specified time."""
        self.spike_matrix = np.zeros((self.params.simulation_steps, self.params.num_neurons), dtype=int)
        self.tot_spike = 0

        for t in range(self.params.simulation_steps):
            self.refractory_timer = np.maximum(0, self.refractory_timer - 1)

            if t % self.params.tau == 0:
                self.stimulate_neuron()

            spiking_neurons = (self.membrane_potentials >= self.params.theta) & (self.refractory_timer == 0)

            self.spike_matrix[t, :] = spiking_neurons.astype(int)
            self.tot_spike += np.sum(spiking_neurons)

            for idx in np.where(spiking_neurons)[0]:
                self.spike_times[idx].append(t)

            self.membrane_potentials[spiking_neurons] = 0
            self.refractory_timer[spiking_neurons] = self.params.t_ref

            self.membrane_potentials += spiking_neurons.astype(float) @ self.synaptic_weights - self.params.leak * self.membrane_potentials

    def get_mean_isi(self):
        """Return the mean inter-spike interval (ISI) for the network."""
        total_inter_spike_intervals = []
        for spike_times in self.spike_times:
            if len(spike_times) > 1:
                isi = np.diff(spike_times)
                total_inter_spike_intervals.extend(isi)
        if total_inter_spike_intervals:
            mean_isi = np.mean(total_inter_spike_intervals)
        else:
            mean_isi = 0
        return mean_isi

    def get_total_spikes(self):
        return self.tot_spike