"""
Runs the complete simulation and shows the 4-panel visualization.
"""

import matplotlib.pyplot as plt

from src.simulator import NetworkAttackSimulation

# Configuration (from Binomial, Poisson, Degradation tables)
# Binomial: p=0.85, n=50 | Poisson: λ=0.5, t=10min → E[N]=5 | Degradation: T₀=100 Mbps
config = {
    "network": {"bandwidth": 100, "buffer_size": 1000, "latency": 10},
    "ids": {"detection_prob": 0.85},
    "attacks": {"rate": 0.5, "packets_per_attack": 50},
    "simulation": {"duration_minutes": 10, "sampling_interval": 1.0},
}

# Run simulation
sim = NetworkAttackSimulation(config)
sim.run(duration_minutes=10, seed=42)

# Get results
results = sim.metrics.get_dataframe()

# Visualize (4-panel plot from PDF)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(results["time"], results["throughput_mbps"], linewidth=2)
axes[0, 0].set_xlabel("Time (seconds)")
axes[0, 0].set_ylabel("Throughput (Mbps)")
axes[0, 0].set_title("Network Throughput")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(results["time"], results["total_detections"], linewidth=2, color="red")
axes[0, 1].set_xlabel("Time (seconds)")
axes[0, 1].set_ylabel("Cumulative Detections")
axes[0, 1].set_title("Attack Detections")
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(
    results["time"], results["buffer_utilization"], linewidth=2, color="orange"
)
axes[1, 0].set_xlabel("Time (seconds)")
axes[1, 0].set_ylabel("Buffer Utilization")
axes[1, 0].set_title("Network Buffer")
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(
    results["time"], results["detection_rate"], linewidth=2, color="green"
)
axes[1, 1].axhline(
    y=config["ids"]["detection_prob"],
    color="red",
    linestyle="--",
    label="Theoretical",
)
axes[1, 1].set_xlabel("Time (seconds)")
axes[1, 1].set_ylabel("Detection Rate")
axes[1, 1].set_title("IDS Performance")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
