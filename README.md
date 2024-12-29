This repository contains the code for the paper "**A Mean-Field Approach to Criticality in Spiking Neural Networks for Reservoir Computing**".    
Simulations are performed to validate the theoretical predictions and analyze the spiking dynamics.

Authors: Alessio Basti, Ruggero Freddi, Francesco Cicala.

## Get started
- Clone this repo or download the code
- Install the requirements.txt by running:
```bash
pip install -r requirements.txt
```

## Run SNN simulation
- Adapt config_snn.yaml with desired parameters
- Run `python run_snn.py`

## Run SOqC
- Adapt config_soqc with desired parameters
- Run `python run_soqc.py`

## Repository Structure

```
.
├── src/
│   ├── __init__.py
│   ├── models.py         # SNN model classes and logic
│   ├── snn_plots.py      # Plotting utilities for the SNN
│   ├── soqc.py           # Logic for run_soqc.py
│   └── utils.py          # Utility functions (e.g., config loader)
├── run_snn.py            # Main script to run SNN simulations
├── run_soqc.py           # Main script to run the SOqC procedure
├── snn_config.yaml       # Configuration file for SNN simulations
├── soqc_config.yaml      # Configuration file for SOqC
├── requirements.txt      # Python dependencies
└── README.md             # Top-level documentation and instructions
```