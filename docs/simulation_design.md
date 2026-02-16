## Simulation Design

This document will describe the architecture and design decisions for the network attack simulation.

### Parameter Sources (from Research Tables)

| Source | Parameter | Values Used |
|--------|-----------|-------------|
| **Binomial** | p (detection_prob) | 0.30, 0.45, 0.55, 0.70, 0.85, 0.90, 0.95 |
| **Binomial** | n (packets_per_attack) | 10, 50 |
| **Poisson** | λ (attack rate/min) | 0.1, 0.5, 1.0 |
| **Poisson** | t (duration) | 10 min |
| **Degradation** | T₀ (bandwidth Mbps) | 100 |
| **Degradation** | α (decay rate) | 0.10, 0.30, 0.50 |

### Components

- Event Manager / SimulationEngine
- Attack Generator
- Network Model
- Intrusion Detection System (IDS)
- Metrics Collector
- Integrated `NetworkAttackSimulation` wrapper

