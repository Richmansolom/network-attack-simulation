# Network Attack Simulation

This project contains the simulation and validation components for the network attack model.

It is a discrete-event simulation with the following main components:

- Clock & Event Manager
- Attack Generator
- Network Model
- Intrusion Detection System (IDS)
- Metrics Collector

Parameters are from the Binomial, Poisson, and Degradation Prediction tables (see `docs/simulation_design.md`).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Run simulation (4-panel plot):**
```bash
python main.py
```

**Run validations (attack generation, IDS detection, throughput degradation):**
```bash
python -m experiments.validation_scripts
```

**Generate theoretical model graphs (Binomial, Poisson, Exponential, Degradation):**
```bash
python -m experiments.generate_graphs
```

**Run Phase 3 analysis pipeline:**
```bash
python phase3_analysis.py
```

**PowerShell (generate synthetic dataset only):**
```powershell
python .\phase3_analysis.py --generate-only
```

**PowerShell (run full pipeline using synthetic data):**
```powershell
python .\phase3_analysis.py --use-synthetic
```

**PowerShell (ordered figure popups):**
- On Windows, figures open one-by-one by default (close each figure window to continue).
- Disable popups and save only:
```powershell
python .\phase3_analysis.py --no-figures
```
- Force popup mode explicitly:
```powershell
python .\phase3_analysis.py --show-figures
```

**Use a custom results CSV path:**
```powershell
python .\phase3_analysis.py --results-file results\all_results.csv
```

**Run tests:**
```bash
pytest
```

On Windows PowerShell, if module imports fail:
```powershell
$env:PYTHONPATH='.'
```

## Project structure

```
network-attack-simulation/
├── src/
│   ├── __init__.py
│   ├── simulator.py
│   ├── network.py
│   ├── ids.py
│   ├── attack_generator.py
│   └── metrics.py
├── tests/
│   ├── test_network.py
│   ├── test_ids.py
│   └── test_attack_generator.py
├── experiments/
│   ├── validation_scripts.py
│   ├── generate_graphs.py
│   └── (additional experiment scripts)
├── data/
│   └── results/
├── results/
├── analysis/ (optional outputs)
├── docs/
│   └── simulation_design.md
├── main.py
├── requirements.txt
└── README.md
```