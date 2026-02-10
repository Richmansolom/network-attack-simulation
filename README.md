# Network Attack Simulation (Phase 2)

This project implements the **Phase 2: Network Attack Simulation** specified in your course handout.  
It is a discrete-event simulation with the following main components:

- Clock & Event Manager
- Attack Generator
- Network Model
- Intrusion Detection System (IDS)
- Metrics Collector

## Project structure

The intended structure is:

- `src/`
  - `__init__.py`
  - `simulator.py`
  - `network.py`
  - `ids.py`
  - `attack_generator.py`
  - `metrics.py`
- `tests/`
  - `test_network.py`
  - `test_ids.py`
  - `test_attack_generator.py`
- `experiments/`
  - `validation_scripts.py`
- `data/`
  - `results/`
- `docs/`
  - `simulation_design.md`

## Setup