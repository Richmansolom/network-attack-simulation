# Network Attack Simulation (Phase 2)

This project implements the **Phase 2: Network Attack Simulation** specified in your course handout.  
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

**Run tests:**
```bash
pytest
```

**Phase 3 experiments:**
```bash
python -m experiments.phase3_experiment_template
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
│   └── validation_scripts.py
├── data/
│   └── results/
├── docs/
│   └── simulation_design.md
├── main.py
├── requirements.txt
└── README.md
```