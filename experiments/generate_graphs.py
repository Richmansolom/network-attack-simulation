"""
Generate the 4 theoretical model graphs from the Research PDF.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def graph1_binomial():
    """Graph 1: P(X≥1) = 1-(1-p)^n vs n for different p values."""
    n = np.arange(1, 26)
    p_values = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]

    fig, ax = plt.subplots(figsize=(10, 6))
    for p in p_values:
        prob = 1 - (1 - p) ** n
        ax.plot(n, prob, linewidth=2, label=f"p = {p}")

    ax.set_xlabel("n (packets per attack)")
    ax.set_ylabel("P(X ≥ 1)")
    ax.set_title("Graph 1: Binomial Detection Model – P(X≥1) vs n")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.show()


def graph2_poisson():
    """Graph 2: Poisson(λ=5) – P(N=k) bar chart, peak at k=5."""
    lam = 5  # E[N] = 5 over 10 min
    k = np.arange(0, 16)
    probs = stats.poisson.pmf(k, lam)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(k, probs, color="#3498db", edgecolor="black")
    ax.axvline(lam, color="red", linestyle="--", linewidth=2, label=f"E[N] = {lam}")
    ax.set_xlabel("k (number of attacks)")
    ax.set_ylabel("P(N = k)")
    ax.set_title("Graph 2: Poisson Distribution – N ~ Poisson(λ=5)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def graph3_exponential():
    """Graph 3: Inter-arrival T ~ Exponential(λ=0.5), mean=2 min, median=1.39 min."""
    lam = 0.5  # attacks per minute
    t = np.linspace(0.01, 8, 200)
    pdf = lam * np.exp(-lam * t)
    mean_val = 1 / lam
    median_val = np.log(2) / lam

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, pdf, "b-", linewidth=2, label="f(t) = λe^(-λt)")
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_val:.2f} min")
    ax.axvline(median_val, color="green", linestyle="--", linewidth=2, label=f"Median = {median_val:.2f} min")
    ax.set_xlabel("t (inter-arrival time, minutes)")
    ax.set_ylabel("Density")
    ax.set_title("Graph 3: Exponential Inter-Arrival – T ~ Exponential(λ=0.5)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    plt.tight_layout()
    plt.show()


def graph4_degradation():
    """Graph 4: Throughput decay T(t) = T₀ × e^(-αt) for α = 0.1, 0.3, 0.5."""
    T0 = 100  # Mbps
    t_min = np.linspace(0, 12, 200)
    alphas = [0.10, 0.30, 0.50]

    fig, ax = plt.subplots(figsize=(10, 6))
    for alpha in alphas:
        T = T0 * np.exp(-alpha * t_min)
        half_life = np.log(2) / alpha
        ax.plot(t_min, T, linewidth=2, label=f"α = {alpha} (t½ = {half_life:.2f} min)")
    ax.set_xlabel("t (time, minutes)")
    ax.set_ylabel("T(t) Mbps")
    ax.set_title("Graph 4: Throughput Degradation – T(t) = T₀ × e^(-αt)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    graph1_binomial()
    graph2_poisson()
    graph3_exponential()
    graph4_degradation()
