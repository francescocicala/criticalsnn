import numpy as np
import matplotlib.pyplot as plt
import random
import math
import tkinter as tk
from tkinter import ttk
import networkx as nx
import pandas as pd

"""NOTA: A RIGA 221 C' E' UN COMMENTO CHE TI DICE FERMATI QUI. 
NON TI INTERESSA QUELLO CHE VIENE DOPO. 
LE NOVITA' SONO TUTTE (SPERO) PRECEDUTE DA UN COMMENTO TRA VIRGOLETTE CHE TE LE SEGNALA.

PER I PARAMETRI IMPRELMENTA I SEGUENTI CONTROLLI

if (tau * N * alpha * theta) != 0 and I / (tau * N * alpha * theta) < 1:
        print("condiczione violate I / (tau * N * alpha * theta) < 1")
        continue
    if (tau * N * theta) != 0 and 2 * I / (tau * N * theta) > 1:
        print("condiczione violate 2 * I / (tau * N * theta) > 1")
        continue
    if alpha != 0 and 1 / (2 * alpha) < 1:
        print("condiczione violate 1 / (2 * alpha) < 1")
        continue
"""

def lzw_complexity_from_matrix(matrix):
    def lzw(seq):
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

    vector = matrix.T.flatten()  
    vector_str = "".join(map(str, vector))  

    complexity = lzw(vector_str)
    return complexity

class NeuronNetwork:
    def __init__(self, N, theta, tau, I, W_mean, sim_time, t_ref, alpha, p=0.1, k=10):
        """QUI CI SONO MODIFICHE DA FARE"""
        self.tot_spike = 0
        """L'ALPHA VIENE ORA SPALMATO SU T_REF CICLI, QUINDI VA DIVISO PER T_REF"""
        self.alpha = alpha / t_ref
        self.N = N
        self.theta = theta
        """TAU VIENE ORA SPALMATO SU T_REF CICLI, QUINDI VA MOLTIPLICATO PER T_REF"""
        self.tau = tau * t_ref
        self.I = I
        self.sim_time = sim_time
        self.t_ref = t_ref  
        self.time_step = 1  
        self.w_aux = W_mean

        self.membrane_potentials = np.random.uniform(0, self.theta, self.N)
        self.spike_times = [[] for _ in range(N)]  
        
        """IMPORTANTE MODIFICA, ORA LA RETE GENENERATA E' UNA SMALL WORLD"""
        self.generate_synaptic_weights(p=p, k=k)

        self.refractory_timer = np.zeros(N)

    def generate_synaptic_weights(self, p=0.1, k=10):
        """
        QUESTA FUNZIONE GENERA LA RETE SMALL WORLD, LA ORIENTA CASULAMENTE E ASSEGNA AGLI ARCHI PESI 
        CASUALI ESTRATTTI DA NORMALE. INFINE, CALCOLA IL GRADO MEDIO D'INGRESSO 
        (NUMERO MEDIO DI NEURONI PRE-SINAPTICI)
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
        """IL GRADO MEDIO E' UN PARAMETRO IMPORTTANTE NELLE NOSTRE FORMULE. CORRISPONDE A beta * N"""
        return self.avg_in_degree

    def stimulate_neuron(self, k):
        """SI DEVONO POTER SIMOLARE k NEURONI ALLA VOLTA"""
        target_neurons = random.sample(range(self.N), k)
        self.membrane_potentials[target_neurons] += self.I

    def simulate(self):
        self.tot_spike = 0
        self.spike_matrix = np.zeros((self.sim_time, self.N), dtype=int)
        """DA QUI SI CALCOLANO I PARAMETRI PER LA NUOVA STIMOLAZIONE"""
        tau_intero = int(self.tau)
        MCD = math.gcd(int(self.tau * 10), 10)
        tau_n = int(self.tau * 10 / MCD)
        tau_d = int(10 / MCD)
        cnt_I = 0
        """QUI HO FINITO DI CALCOLARE I PARAMETRI PER LA NUOVA STIMOLAZIONE"""
        for t in range(self.sim_time):
            self.refractory_timer = np.maximum(0, self.refractory_timer - self.time_step)
            """QUI INIZIA LA LOGICA PER LA NUOVA STIMOLAZIONE"""
            if self.tau < 1:
                if cnt_I % tau_n == 0:
                    self.stimulate_neuron(tau_d)
            if self.tau >= 1:
                if tau_intero > 0 and cnt_I % tau_intero == 0:
                    self.stimulate_neuron(1)
                if cnt_I == self.tau * 10:
                    cnt_I = 0
            cnt_I += 1
            """QUI FINISCE LA LOGICA PER LA NUOVA STIMOLAZIONE"""
            spiking_neurons = (self.membrane_potentials >= self.theta) & (self.refractory_timer == 0)
            self.spike_matrix[t, :] = spiking_neurons.astype(int)
            self.tot_spike += np.sum(spiking_neurons)

            for idx in np.where(spiking_neurons)[0]:
                self.spike_times[idx].append(t)

            self.membrane_potentials[spiking_neurons] = 0
            """QUEL + 1 SUL T_REF MIGLIORA MOLTO LE COSE, CREDO DEBBA ESSERE MESSO ANCHE NELLA TUA VERSIONE"""
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
        return float(mean_isi) / self.t_ref
    

def calculate_W_avg( delta_avg, alpha, theta, I, tau, N, tau_ref, beta):
        """
        QUESTA FUUNZIONE CALCOLA IL W TEORICO A PARTIRE DA UN ISI.

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
    
def run_simulation(N, theta, tau, I, alpha, sim_time, iterations, p, k):
    """DI QUESTTA A TE INTERESSA IL GIUSTO. L COSE INTERESSANTI PER TE LE METTERO IN EVIDENZA"""

    """QUESTI SONO PARAMETRI CHE TI SERVONO DOPO"""
    k = int(N * k)
    beta = int (k / 2) 
    w_critical = theta / beta  - (2 * I) / (tau * beta * N)

    """BENCHE' I SEGUETI VALORI NON SIANO SCRITTI NELLA ROCCIA DOPO MOLTE PROVE MI SEMBRA CHE 
    QUESTA SIA LA CONFIGURAZIONE MIGLIORE PER I GRAFICI: 200 PUNTI PER CURVA SU DI 
    UN INTERVALLO CHE VA DA w_critical - w_critical * 0.5 A w_critical + w_critical * 15.
    NOTA: PER QUESTIONI GRAFICHE IO POI GRAFICAVO TUTTO L'INTERVALLO SOLO PER L'ISI. 
    PER SPIKE E LZW USAVO I PRIMI 100 VALORI."""
    steps = 400
    start = w_critical - w_critical * 0.5
    end = w_critical + w_critical * 15
    step = (end - start) / steps

    w_means, num_spikes, mean_isis, lzw_complexities = [], [], [], []

    for i in range(iterations):
        cnt = 0
        for idx, w_mean in enumerate(np.arange(start, end, step)):
            cnt = cnt + 1
            """IL COSTRUTTTORE DELLA CLASSE NeuronNetwork PREDE IN PIU' I PARAMETRI PER LA SMALL WORLD"""
            network = NeuronNetwork(N, theta, tau, I, w_mean, sim_time, t_ref=10, alpha=alpha, p = p, k = k)
            beta = network.in_degree()
            """DA QUI IN POI A TE NON INTERESSA PIU'. 
            IL RESTO SERE SOLO PER CALCOLARE LE METRICHE DI ERRORE """
            if cnt % 10 ==0:
                print(cnt)
            spike_matrix = network.simulate()
            lzw_complexity = lzw_complexity_from_matrix(spike_matrix)
            total_spikes = network.tot_spike
            mean_isi = network.calculate_mean_isi()
            w_means.append(w_mean)
            num_spikes.append(total_spikes)
            mean_isis.append(mean_isi)
            lzw_complexities.append(lzw_complexity)
    w_critical = theta / beta  - (2 * I) / (tau * beta * N)

    punto_critico = w_critical

    w_means = np.array(w_means)
    lzw_complexities = np.array(lzw_complexities)

    subset_length = int(steps / 3)
    w_means_subset = w_means[:subset_length]
    lzw_complexities_subset = lzw_complexities[:subset_length]

    indice_max = np.argmax(lzw_complexities_subset)
    w_mean_max = w_means_subset[indice_max]

    distanza = abs(w_mean_max - punto_critico)

    calculated_w = calculate_W_avg(
        delta_avg=mean_isis, 
        alpha=alpha,  
        theta=theta,  
        I=I,  
        tau=tau, 
        N=N,  
        tau_ref=1, 
        beta = beta
    )

    # Verifica le dimensioni degli array
    if w_means.shape != calculated_w.shape:
        raise ValueError("Gli array 'w_means' e 'calculated_w' devono avere la stessa forma.")

    # Calcola la differenza assoluta
    differenza = np.abs((w_means - calculated_w)/w_means)

    # Calcola la differenza media
    differenza_media = np.mean(differenza)

    print(f"Differenza Media: {differenza_media:.4f}")

    return distanza, distanza / w_mean_max, w_mean_max, punto_critico
print("Fine controlli. Inizio simulazione.")
grid_search = [
    (1000, 14, 0.01, 1.5, 0.01, 10000, 1, 0.1, 0.2), 
    (1000, 14, 0.01, 1.5, 0.01, 10000, 1, 0.25, 0.2), 
    (1000, 1.4, 0.01, 1.5, 0.01, 10000, 1, 0.1, 0.2), 
    (1000, 1.4, 0.01, 1.5, 0.01, 10000, 1, 0.25, 0.2), 
    (1000, 14, 0.1, 1.5, 0.001, 10000, 1, 0.1, 0.2), 
    (1000, 14, 0.1, 1.5, 0.001, 10000, 1, 0.25, 0.2), 
    (1000, 14, 9.99, 1.5, 0.00001, 30000, 1, 0.1, 0.2), 
    (1000, 14, 9.99, 1.5, 0.00001, 30000, 1, 0.25, 0.2), 
    (1000, 14, 0.01, 1.5, 0.0001, 10000, 1, 0.1, 0.2), 
    (1000, 14, 0.01, 1.5, 0.0001, 10000, 1, 0.25, 0.2), 
    (1000, 1.4, 0.01, 1.5, 0.1, 10000, 1, 0.1, 0.2), 
    (1000, 1.4, 0.01, 1.5, 0.1, 10000, 1, 0.25, 0.2), 
    ]
grid_search = [
    (1000, 14, 0.01, 1.5, 0.01, 10000, 1, 0.1, 0.2), 
]
for parameters in grid_search:
    N, theta, tau, I, alpha, sim_time, iterations, k, p  = parameters
    if (tau * N * alpha * theta) != 0 and I / (tau * N * alpha * theta) < 1:
        print("condiczione violate I / (tau * N * alpha * theta) < 1")
        continue
    if (tau * N * theta) != 0 and 2 * I / (tau * N * theta) > 1:
        print("condiczione violate 2 * I / (tau * N * theta) > 1")
        continue
    if alpha != 0 and 1 / (2 * alpha) < 1:
        print("condiczione violate 1 / (2 * alpha) < 1")
        continue
results =[]
for parameters in grid_search:    
    N, theta, tau, I, alpha, sim_time, iterations, k, p  = parameters
    distanza, distanza_perc, massimo_lzw, punto_critico = run_simulation(N, theta, tau, I, alpha, sim_time, iterations, p, k)
    results.append((N, theta, tau, I, alpha, sim_time, iterations, k, p, massimo_lzw, punto_critico, distanza, distanza_perc, I / (tau * N * alpha * theta),2 * I / (tau * N * theta),  1 / (2 * alpha)))
column_names = [
    'N', 'theta', 'tau', 'I', 'alpha', 
    'sim_time', 'iterations', 'k', 'p', 'massimo_lzw', 'punto_critico',
    'distanza', 'dist_percentuale', 
    'cond1', 'cond2', 'cond3'
]

df = pd.DataFrame(results, columns=column_names)

print(df)  

df.to_csv('search_grid.csv', index=False)

print("Il DataFrame è stato salvato con successo in 'search_grid.csv'")


