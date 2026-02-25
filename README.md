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

- `src/` – simulator, network, IDS, attack generator, metrics
- `tests/` – unit tests (`conftest.py` for import path)
- `experiments/` – validation_scripts, generate_graphs, phase3_experiment_template
- `data/results/` – output files
- `docs/` – simulation_design.md