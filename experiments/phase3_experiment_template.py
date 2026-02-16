"""
Phase 3 Experiment Template (Section 7.2)
Modify for each research question.
"""

import pandas as pd

from src.simulator import NetworkAttackSimulation

# Base configuration (Degradation table: T₀=100, Binomial n=50, Poisson t=10)
config = {
    "network": {"bandwidth": 100, "buffer_size": 1000, "latency": 10},
    "ids": {"detection_prob": 0.85},
    "attacks": {"rate": 0.5, "packets_per_attack": 50},
    "simulation": {"duration_minutes": 10, "sampling_interval": 1.0},
}

# Configuration (Binomial p values, Poisson λ values from tables)
EXPERIMENT_NAME = "detection_vs_load"
N_REPLICATIONS = 30
DETECTION_PROBS = [0.30, 0.45, 0.55, 0.70, 0.85, 0.90, 0.95]  # Binomial table
ATTACK_RATES = [0.1, 0.5, 1.0]  # Poisson: λ=0.1→μ=1, λ=0.5→μ=5, λ=1.0→μ=10 @ t=10min

# Run experiments
all_results = []
for p in DETECTION_PROBS:
    for lambda_rate in ATTACK_RATES:
        for rep in range(N_REPLICATIONS):
            config["ids"]["detection_prob"] = p
            config["attacks"]["rate"] = lambda_rate
            sim = NetworkAttackSimulation(config)
            sim.run(duration_minutes=10, seed=1000 + len(all_results))
            summary = sim.metrics.get_summary_stats()
            summary["p"] = p
            summary["lambda"] = lambda_rate
            summary["rep"] = rep
            all_results.append(summary)

# Save results
df = pd.DataFrame(all_results)
df.to_csv(f"{EXPERIMENT_NAME}_results.csv", index=False)

# Analyze
grouped = df.groupby(["p", "lambda"])
print(grouped["final_detection_rate"].mean())
