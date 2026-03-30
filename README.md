# Network Attack Simulation (Phase 2 + Phase 3)

This project implements the **Phase 2 (simulation)** and **Phase 3 (experimental/statistical analysis)**
requirements from your course handout.

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

**Run full Phase 3 pipeline (plan + experiments + analysis):**
```bash
python phase3_analysis.py
```

Phase 3 writes **guide-aligned figures** under `analysis/figures/`:

| File | Hypothesis |
|------|------------|
| `h1_detection_overlay.png` | H1 — binomial P(detect ≥ 1) vs theory |
| `h2_poisson_overlay.png` | H2 — Poisson attack counts |
| `h3_exponential_analysis.png` | H3 — exponential inter-arrivals |
| `h4_degradation_analysis.png` | H4 — throughput decay |
| `h5_error_heatmap.png` | H5 — prediction error heatmap |
| `h5_interaction_plot.png` | H5 — detection vs λ by p |
| `h1_forest_plot.png` | H1 — forest plot (observed vs theory, CIs) |

Tables are under `analysis/tables/` (e.g. `h1_results.csv`, …).

**Standard design:** `5 p × 4 λ × 3 α × 4 n = 240` conditions × `30` = **7,200** runs. Default **`p`** for H1: **`[0.30, 0.50, 0.70, 0.85, 0.95]`** (`GUIDE_FACTORIAL_DETECTION_PROBS`; range **0.30–0.95**, not 0.70–0.95). Override `detection_probs` only if your handout differs.

On Windows PowerShell, if `src` import errors appear for module-style commands:
```powershell
$env:PYTHONPATH='.'
python "experiments/phase3_experiment_template.py"
```

## GitHub Actions artifacts

The GitHub Actions workflow runs tests, generates Phase 3 synthetic outputs, and uploads artifacts.

Artifact name:
- `phase3-results-and-figures`

Included files:
- `results/*.csv`
- `analysis/tables/*.csv`
- `analysis/figures/*.png`

How to download:
1. Open your repository on GitHub.
2. Go to **Actions** and open a completed workflow run.
3. Scroll to **Artifacts** and download `phase3-results-and-figures`.

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
│   └── phase3_experiment_template.py
├── data/
│   └── results/
├── results/
├── analysis/
├── docs/
│   └── simulation_design.md
├── main.py
├── phase3_analysis.py
├── requirements.txt
└── README.md
```