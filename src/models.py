import numpy as np
import random
from dataclasses import dataclass, field

@dataclass
class SNNParameters:
    """
    Parameters for the Spiking Neural Network (SNN).
    """
    N: int = 1000  # Number of neurons
    theta: float = 7.0  # Firing threshold
    tau: float = 10.0  # Time interval for external current
    I: float = 0.5  # Intensity of external current
    sim_time: int = 20000  # Simulation time steps
    t_ref: float = 2.0  # Refractory period
    leak: float = 0.  # Leak parameter for leaky integrate-and-fire
    W_mean: float = 0.0  # Initial mean synaptic weight (will be updated)
    std_dev: float = 500 # std dev parameter for the synaptic weights
    w_min: float = 0.006 # minimum value for the mean synaptic weight
    w_max: float = 0.25  # maximum value for the mean synaptic weight
    w_samples: int = 500 # number of samples for the mean synaptic weight
    w_iterations: int = 10 # number of iterations

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
    time_step: float = 1.0
    tot_spike: int = 0

    def __post_init__(self):
        self.membrane_potentials = np.random.uniform(0, self.params.theta, self.params.N)
        self.synaptic_weights = np.random.normal(loc=self.params.W_mean, scale=self.params.W_mean / self.params.std_dev, size=(self.params.N, self.params.N))
        
        self.spike_times = [[] for _ in range(self.params.N)]
        np.fill_diagonal(self.synaptic_weights, 0)
        self.refractory_timer = np.zeros(self.params.N)

    def update_synaptic_weights(self, new_w_mean):
        self.synaptic_weights = np.random.normal(loc=new_w_mean, scale=new_w_mean/self.params.std_dev, size=(self.params.N, self.params.N))
        np.fill_diagonal(self.synaptic_weights, 0)

    def stimulate_neuron(self):
        """Deliver external current to a random neuron."""
        target_neuron = random.randint(0, self.params.N - 1)
        self.membrane_potentials[target_neuron] += self.params.I

    def simulate(self):
        """Simulate the network's activity over the specified time."""
        self.spike_matrix = np.zeros((self.params.sim_time, self.params.N), dtype=int)
        self.tot_spike = 0

        for t in range(self.params.sim_time):
            self.refractory_timer = np.maximum(0, self.refractory_timer - self.time_step)

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
        return self.tot_spike

    def calculate_mean_isi(self):
        """Calculate the mean inter-spike interval (ISI) for the network."""
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

    def get_mean_synaptic_weight(self):
        """Return the average synaptic weight."""
        return np.mean(self.synaptic_weights)