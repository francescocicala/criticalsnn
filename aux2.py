import numpy as np
import matplotlib.pyplot as plt
import random
import math
import tkinter as tk
from tkinter import ttk
import networkx as nx

def lzw_complexity_from_matrix(matrix):
    """
    Calculate the Lempel-Ziv (LZW) complexity of a vector created by concatenating the columns of a matrix.
    """
    def lzw(seq):
        """Calculate the LZW complexity of a binary sequence."""
        dictionary = {}
        w = ""
        complexity = 0
        for c in seq:
            wc = w + c
            if wc not in dictionary:
                dictionary[wc] = len(dictionary)
                complexity += 1
                w = c
            else:
                w = wc
        return complexity

    # Transform the matrix into a vector by concatenating its columns
    vector = matrix.T.flatten()  # Transpose and then flatten to read column by column
    vector_str = "".join(map(str, vector))  # Convert the vector into a string

    # Calculate the LZW complexity of the concatenated vector
    complexity = lzw(vector_str)
    return complexity

class NeuronNetwork:
    def __init__(self, N, theta, tau, I, W_mean, sim_time, t_ref, alpha, p=0.1, k=10):
        """
        Inizializza la rete neurale.

        Parametri:
        - N (int): Numero di neuroni.
        - theta (float): Soglia di attivazione.
        - tau (float): Costante di tempo.
        - I (float): Corrente di input.
        - W_mean (float): Peso sinaptico medio.
        - sim_time (int): Tempo di simulazione.
        - t_ref (float): Periodo refrattivo.
        - alpha (float): Tempo caratteristico reciproco.
        - p (float, opzionale): Probabilità di rimescolamento per la rete Small-World. Default 0.1.
        - k (int, opzionale): Numero di vicini per nodo nella rete Small-World. Default 10.
        """
        
        # Stampa dei parametri (opzionale, commentata per evitare output durante l'esecuzione)
        """
        print(f"Initializing Neuron Network with the following parameters:")
        print(f"  N (Number of neurons): {N}")
        print(f"  θ (Activation threshold): {theta}")
        print(f"  τ (Time constant): {tau}")
        print(f"  I (Input current): {I}")
        print(f"  W_mean (Mean synaptic weight): {W_mean}")
        print(f"  Simulation time: {sim_time}")
        print(f"  t_ref (Refractory period): {t_ref}")
        print(f"  α (Reciprocal characteristic time): {alpha}")
        print(f"  p (Rewiring probability): {p}")
        print(f"  k (Number of neighbors): {k}")
        """

        # Inizializzazione delle variabili di istanza
        self.tot_spike = 0
        self.alpha = alpha / t_ref
        self.N = N
        self.theta = theta
        self.tau = tau * t_ref
        self.I = I
        self.sim_time = sim_time
        self.t_ref = t_ref  # Periodo refrattivo
        self.time_step = 1  # Time step per la simulazione
        self.w_aux = W_mean

        # Inizializzazione dei potenziali di membrana uniformemente tra 0 e theta
        self.membrane_potentials = np.random.uniform(0, self.theta, self.N)
        self.spike_times = [[] for _ in range(N)]  # Memorizza i tempi di spike per ogni neurone

        # Generazione della matrice delle connessioni sinaptiche utilizzando il metodo separato
        self.generate_synaptic_weights(p=p, k=k)

        # Inizializzazione del timer di refrattività per ogni neurone
        self.refractory_timer = np.zeros(N)

    def generate_synaptic_weights(self, p=0.1, k=10):
        """
        Genera la matrice delle connessioni sinaptiche utilizzando una rete Small-World.

        Parametri:
        - p (float): Probabilità di rimescolamento per la rete Small-World.
        - k (int): Numero di vicini per nodo nella rete Small-World.
        """
        # Genera la rete Small-World non orientata utilizzando il modello di Watts-Strogatz
        small_world_graph = nx.watts_strogatz_graph(n=self.N, k=k, p=p, seed=None)

        # Inizializza la matrice delle connessioni con zeri
        synaptic_weights = np.zeros((self.N, self.N))

        # Orienta casualmente gli archi e assegna i pesi
        for edge in small_world_graph.edges():
            i, j = edge
            if np.random.rand() < 0.5:
                # Orienta l'arco come i -> j
                synaptic_weights[i, j] = np.random.normal(loc=self.w_aux, scale=abs(self.w_aux) / 10)
            else:
                # Orienta l'arco come j -> i
                synaptic_weights[j, i] = np.random.normal(loc=self.w_aux, scale=abs(self.w_aux) / 10)

        # Assicura che non ci siano autoconnessioni
        np.fill_diagonal(synaptic_weights, 0)

        # Assegna la matrice delle connessioni all'istanza
        self.synaptic_weights = synaptic_weights
        
        # Calcola il grado di ingresso per ogni nodo
        # Il grado di ingresso è il numero di connessioni entranti per ciascun nodo
        in_degrees = np.count_nonzero(self.synaptic_weights, axis=0)  # Conta gli archi entranti per ogni nodo
        self.avg_in_degree = in_degrees.mean()  # Calcola la media del grado di ingresso

    def in_degree(self):
        return self.avg_in_degree

    def stimulate_neuron(self, k):
        """Deliver external current to k random unique neurons."""
        target_neurons = random.sample(range(self.N), k)
        self.membrane_potentials[target_neurons] += self.I

    def simulate(self):
        """Simulate the network's evolution over time."""
        self.tot_spike = 0
        self.spike_matrix = np.zeros((self.sim_time, self.N), dtype=int)
        tau_intero = int(self.tau)
        MCD = math.gcd(int(self.tau * 10), 10)
        tau_n = int(self.tau * 10 / MCD)
        tau_d = int(10 / MCD)
        cnt_I = 0
        for t in range(self.sim_time):
            self.refractory_timer = np.maximum(0, self.refractory_timer - self.time_step)
            if self.tau < 1:
                if cnt_I % tau_n == 0:
                    self.stimulate_neuron(tau_d)
            if self.tau >= 1:
                if tau_intero > 0 and cnt_I % tau_intero == 0:
                    self.stimulate_neuron(1)
                if cnt_I == self.tau * 10:
                    cnt_I = 0
            cnt_I += 1

            spiking_neurons = (self.membrane_potentials >= self.theta) & (self.refractory_timer == 0)
            self.spike_matrix[t, :] = spiking_neurons.astype(int)
            self.tot_spike += np.sum(spiking_neurons)

            for idx in np.where(spiking_neurons)[0]:
                self.spike_times[idx].append(t)

            self.membrane_potentials[spiking_neurons] = 0
            self.refractory_timer[spiking_neurons] = self.t_ref + 1
            self.membrane_potentials = (1 - self.alpha) * self.membrane_potentials + spiking_neurons.astype(float) @ self.synaptic_weights
        return self.spike_matrix
    
    def calculate_mean_isi(self):
        total_inter_spike_intervals = []
        for spike_times in self.spike_times:
            if len(spike_times) > 1:
                isi = np.diff(spike_times)
                total_inter_spike_intervals.extend(isi)
        if total_inter_spike_intervals:
            mean_isi = np.mean(total_inter_spike_intervals)
        else:
            mean_isi = 0
        print("W", self.w_aux, " isi: ", mean_isi)
        return float(mean_isi) / self.t_ref

def run_simulation_with_progress(N, theta, tau, I, alpha, sim_time, iterations, progress_iter, progress_w_mean):
    beta = 10
    w_critical = theta / beta  - (2 * I) / (tau * beta * N)
    steps = 200
    start = w_critical - w_critical * 0.5
    end = w_critical + w_critical * 15
    step = (end - start) / steps

    w_means, num_spikes, mean_isis, lzw_complexities = [], [], [], []

    for i in range(iterations):
        progress_iter["value"] = (i + 1) / iterations * 100
        progress_iter.update()
        for idx, w_mean in enumerate(np.arange(start, end, step)):
            progress_w_mean["value"] = (idx + 1) / steps * 100
            progress_w_mean.update()
            network = NeuronNetwork(N, theta, tau, I, w_mean, sim_time, t_ref=10, alpha=alpha, p = 0.2, k = 20)
            beta = network.in_degree()
            print("beta: ", beta)
            spike_matrix = network.simulate()
            lzw_complexity = lzw_complexity_from_matrix(spike_matrix)
            total_spikes = network.tot_spike
            mean_isi = network.calculate_mean_isi()
            w_means.append(w_mean)
            num_spikes.append(total_spikes)
            mean_isis.append(mean_isi)
            lzw_complexities.append(lzw_complexity)
    w_critical = theta / beta  - (2 * I) / (tau * beta * N)


    def calculate_W_avg(delta_avg, alpha, theta, I, tau, N, tau_ref, beta):
        """
        Calcola <W>(<Δ>) secondo l'equazione:

        <W>(<Δ>) = [α θ <Δ>^2 / (1 - exp(-alpha * <Delta>)) - (I <Delta>^2) / (tau * N)] / [(N - 1)(<Delta> - tau_ref)]

        Parametri:
        - delta_avg (float o array-like): Valore medio di Δ (⟨Δ⟩).
        - alpha (float): Valore di α.
        - theta (float): Valore di θ.
        - I (float): Valore di I.
        - tau (float): Valore di τ.
        - N (int): Valore di N (deve essere maggiore di 1).
        - tau_ref (float): Valore di τ_ref.

        Ritorna:
        - W_avg (float o array-like): Valore calcolato di <W>(<Δ>).
        """
        # Converti delta_avg in un array numpy per supportare input scalari e vettoriali
        delta_avg_1 = np.array(delta_avg, dtype=np.float64)
        for ele in delta_avg_1:
            print(ele)        
        # Calcola il denominatore (1 - exp(-alpha * delta_avg))
        denominator_exp = 1 - np.exp(-alpha * delta_avg_1)
        
        # Evita divisione per zero o valori troppo piccoli
        epsilon = 1e-10
        denominator_exp = np.where(denominator_exp < epsilon, epsilon, denominator_exp)
        
        # Primo termine del numeratore: alpha * theta * delta_avg^2 / (1 - exp(-alpha * delta_avg))
        term1 = (alpha * theta * delta_avg_1**2) / denominator_exp
        
        # Secondo termine del numeratore: (I * delta_avg^2) / (tau * N)
        term2 = (I * delta_avg_1**2) / (tau * N)
        
        # Numeratore complessivo
        numerator = term1 - term2
        
        # Denominatore complessivo: (N - 1) * (delta_avg - tau_ref)
        denominator = beta * (delta_avg_1 - tau_ref)
        
        # Evita divisione per zero nel denominatore
        denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
        
        # Calcola <W>(<Δ>)
        W_avg = numerator / denominator
        
        return W_avg

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    punto_critico = w_critical

    # Primo grafico: Numero di Spike
    axes[0, 1].plot(w_means[:int(steps/3)], num_spikes[:int(steps/3)], color='blue', alpha=0.7,linewidth=2)
    axes[0, 1].set_xscale('linear')
    axes[0, 1].set_yscale('linear')
    axes[0, 1].axvline(x=punto_critico, color='red', linestyle='--', label='w_critical')
    axes[0, 1].legend()
    axes[0, 1].set_title("Numero di Spike (Log-Log)")

    # Secondo grafico: ISI Medio
    axes[0, 0].plot(mean_isis, w_means, color='green', alpha=0.7, label="Data",linewidth=2)
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].axvline(x=punto_critico, color='red', linestyle='--', label='w_critical')

    # Calcolo della curva da aggiungere
    calculated_w = calculate_W_avg(
        delta_avg=mean_isis,  # Array degli ISI medi
        alpha=alpha,  # Parametro α
        theta=theta,  # Parametro θ
        I=I,  # Corrente di input
        tau=tau,  # Parametro τ
        N=N,  # Numero di neuroni
        tau_ref=1,  # τ_ref (valore arbitrario, può essere modificato)
        beta = beta
    )

    print("isi: ", np.min(mean_isis), np.max(mean_isis))
    print("W teo: ", np.min(calculated_w), np.max(calculated_w))
    print("W reale: ", np.min(w_means), np.max(w_means))


    axes[0, 0].plot(mean_isis, calculated_w , color='blue', linestyle='--', label="Calculated W_avg con leak")
    axes[0, 0].legend()
    axes[0, 0].set_title("ISI Medio (Log-Log)")


    def calculate_theoretical_isi(w_mean, N, theta, tau, I, t_ref):
        # Converti w_mean in un array NumPy se non lo è già
        w_mean1 = np.array(w_mean)
        
        term1 = theta - w_mean1 * beta
        term2 = np.sqrt(term1**2 + (4 * I * w_mean1 * beta * t_ref) / (tau * N))
        return (tau * N) / (2 * I) * (term1 + term2)

    # Calcolo della curva da aggiungere
    calculated_isi = calculate_theoretical_isi(
        w_mean=w_means,  # Array degli ISI medi
        N=N,              # Numero di neuroni
        theta=theta,      # Parametro θ
        tau=tau,          # Parametro τ
        I=I,              # Corrente di input
        t_ref=1           # τ_ref (valore arbitrario, può essere modificato)
    )

    # Aggiungere la curva al grafico
    axes[0, 0].plot(calculated_isi, w_means  , color='orange', linestyle='--', label="Calculated W_avg", linewidth=2)
    axes[0, 0].legend()
    axes[0, 0].set_title("ISI Medio (Log-Log)")
    
    # Terzo grafico: LZW Complexity
    axes[1, 0].plot(w_means[:int(steps/3)], lzw_complexities[:int(steps/3)], color='red', alpha=0.7, linewidth=2)
    axes[1, 0].axvline(x=punto_critico, color='red', linestyle='--', label='w_critical')
    axes[1, 0].legend()
    axes[1, 0].set_title("LZW Complexity")

    # Rimuovi la cella vuota (seconda riga, seconda colonna)
    fig.delaxes(axes[1, 1])  # Rimuove l'asse in posizione (1, 1)

    plt.tight_layout()
    plt.show()



def create_gui():
    root = tk.Tk()
    root.title("Neural Network Simulation")
    root.geometry("330x540")

    params = {
        "N": tk.IntVar(value=100),
        "theta": tk.DoubleVar(value=140),
        "I": tk.DoubleVar(value=1.5),
        "tau": tk.DoubleVar(value=0.01),
        "alpha": tk.DoubleVar(value=1 / 100),
        "sim_time": tk.IntVar(value=10000),
        "iterations": tk.IntVar(value=1),
    }

    spinboxes = {}

    def validate_parameters():
        errors = []
        N = params["N"].get()
        theta = params["theta"].get()
        I = params["I"].get()
        tau = params["tau"].get()
        alpha = params["alpha"].get()
        sim_time = params["sim_time"].get()

        if (tau * N * alpha * theta) != 0 and I / (tau * N * alpha * theta) < 1:
            errors.append("Condition 1 violated: I / (τNθα) > 1")

        if (tau * N * theta) != 0 and 2 * I / (tau * N * theta) > 1:
            errors.append("Condition 2 violated: 2I / (τNθ) < 1")

        if alpha != 0 and 1 / (2 * alpha) < 1:
            errors.append("Condition 3 violated: α < 1 / (2τ)")

        error_label.config(text="\n".join(errors))
        return errors

    def validate_spinbox(value, param_name):
        """Validate input for a spinbox."""
        try:
            if param_name in ["N", "sim_time", "iterations"]:  # Integer inputs
                value = int(value)
            else:  # Float inputs
                value = float(value)
        except ValueError:
            return False
        
        if param_name == 'tau':
            if round(value,5) != round(value,2):
                return False

        limits = spinbox_limits[param_name]
        return limits[0] <= value <= limits[1]
    
    def on_parameter_change():
        validate_parameters()

    def disable_spinboxes():
        """Disable all spinboxes to prevent further interaction."""
        for spinbox in spinboxes.values():
            spinbox.config(state="disabled")

    def run_and_close():
        errors = validate_parameters()
        if not errors:
            disable_spinboxes()
            run_simulation_with_progress(
                params["N"].get(),
                params["theta"].get(),
                params["tau"].get(),
                params["I"].get(),
                params["alpha"].get(),
                params["sim_time"].get(),
                params["iterations"].get(),
                progress_iter,
                progress_w_mean,
            )

    spinbox_limits = {
        "N": (100, 10000),
        "theta": (0.1, 1000),
        "I": (0.01, 100),
        "tau": (0.01, 9.99),
        "alpha": (0.000001, 1),
        "sim_time": (10000, 1000000),
        "iterations": (1, 100),
    }

    spinbox_increments = {
        "N": 10,
        "theta": 1,
        "I": 0.01,
        "tau": 0.01,
        "alpha": 0.000001,
        "sim_time": 1000,
        "iterations": 1,
    }

    # Frame per i parametri di input
    frame_inputs = tk.Frame(root)
    frame_inputs.pack(padx=10, pady=10)

    for i, (label, var) in enumerate(params.items()):
        tk.Label(frame_inputs, text=label.capitalize()).grid(row=i, column=0, padx=5, pady=5, sticky="w")
        limits = spinbox_limits[label]
        increment = spinbox_increments[label]
        spinbox = tk.Spinbox(
            frame_inputs,
            from_=limits[0],
            to=limits[1],
            increment=increment,
            textvariable=var,
            width=10,
            validate="key",
            validatecommand=(root.register(lambda value, param=label: validate_spinbox(value, param)), "%P"),
            command=on_parameter_change,  # Call on_parameter_change when interacting
        )
        spinbox.grid(row=i, column=1, padx=5, pady=5)
        spinboxes[label] = spinbox

    # Frame per i messaggi di errore
    error_frame = tk.LabelFrame(root, text="Error Messages", fg="red")
    error_frame.pack(fill=tk.BOTH, padx=30, pady=5)

    error_label = tk.Label(error_frame, text="", fg="red", justify="left", anchor="nw", wraplength=400, font=("Helvetica", 10, "bold"))
    error_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    tk.Button(root, text="Run Simulation", command=run_and_close, bg="green", fg="black", font=("Helvetica", 12, "bold")).pack(pady=10)

    progress_iter = ttk.Progressbar(root, length=250)
    progress_iter.pack(pady=15)
    tk.Label(root, text="Progress: Iterations").pack()

    progress_w_mean = ttk.Progressbar(root, length=250)
    progress_w_mean.pack(pady=15)
    tk.Label(root, text="Progress: w_mean").pack()

    root.mainloop()


if __name__ == "__main__":
    create_gui()
