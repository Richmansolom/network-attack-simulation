import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

from src.attack_generator import AttackGenerator
from src.ids import IntrusionDetectionSystem
from src.simulator import NetworkAttackSimulation


def validate_attack_generation():
    """Validation 1: Attack Generation (Section 6.1) - Poisson: λ=0.5"""
    np.random.seed(42)  # Reproducible results for statistical tests
    lambda_rate = 0.5  # Poisson table: λ=0.5, E[N]=μ=λt
    duration = 1000
    n_trials = 30
    attack_counts = []
    all_inter_arrivals = []

    for _ in range(n_trials):
        gen = AttackGenerator(attack_rate=lambda_rate, packets_per_attack=50)  # Binomial n=50
        attack_times = gen.generate_attack_times(0.0, duration)
        attack_counts.append(len(attack_times))
        if len(attack_times) > 1:
            all_inter_arrivals.extend(np.diff(attack_times))

    # Test counts
    expected_count = lambda_rate * duration
    t_stat, p_value = stats.ttest_1samp(attack_counts, expected_count)
    print("Attack Counts Test:")
    print(f" Expected: {expected_count:.1f}")
    print(f" Observed: {np.mean(attack_counts):.1f}")
    print(f" p-value: {p_value:.4f}")

    # Test inter-arrivals
    ks_stat, ks_pvalue = stats.kstest(all_inter_arrivals, "expon", args=(0, 1 / lambda_rate))
    print("\nInter-Arrival Times Test:")
    print(f" K-S statistic: {ks_stat:.4f}")
    print(f" p-value: {ks_pvalue:.4f}")

    if p_value > 0.05 and ks_pvalue > 0.05:
        print("\nPASS: Both tests pass - attack generation is correct")
    else:
        print("\nFAIL: Tests fail - investigate!")

    # Visualization (Section 3.2.1): Histogram + Q-Q plot
    np.random.seed(42)
    gen = AttackGenerator(attack_rate=lambda_rate, packets_per_attack=50)
    attack_times = gen.generate_attack_times(0.0, 1000.0)
    inter_arrivals = np.diff(attack_times)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Histogram with theoretical overlay
    axes[0].hist(inter_arrivals, bins=50, density=True, alpha=0.7, edgecolor="black")
    x = np.linspace(0, max(inter_arrivals), 100)
    axes[0].plot(
        x,
        lambda_rate * np.exp(-lambda_rate * x),
        "r-",
        linewidth=2,
        label="Theoretical",
    )
    axes[0].set_xlabel("Inter-Arrival Time (minutes)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Inter-Arrival Time Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # Q-Q plot
    stats.probplot(
        inter_arrivals, dist=stats.expon(scale=1 / lambda_rate), plot=axes[1]
    )
    axes[1].set_title("Q-Q Plot vs Exponential")
    plt.tight_layout()
    plt.show()


def validate_ids_detection():
    """Validation 2: Detection Accuracy (Section 4.2.1)"""
    import random

    random.seed(42)

    # Test parameters (Binomial table: p=0.85, n=50)
    p_theoretical = 0.85
    n_packets = 50
    n_trials = 30
    detection_rates = []

    for _ in range(n_trials):
        ids = IntrusionDetectionSystem(detection_probability=p_theoretical)
        for i in range(n_packets):
            packet = {"id": i, "malicious": True, "size_kb": 1}
            ids.inspect_packet(packet)
        detection_rates.append(ids.get_detection_rate())

    detection_rates = np.array(detection_rates)

    # Statistical analysis
    mean_rate = np.mean(detection_rates)
    std_rate = np.std(detection_rates, ddof=1)
    se_rate = std_rate / np.sqrt(n_trials)
    ci_lower = mean_rate - 1.96 * se_rate
    ci_upper = mean_rate + 1.96 * se_rate

    # One-sample t-test
    t_stat, p_value = stats.ttest_1samp(detection_rates, p_theoretical)
    print(f"Mean detection rate: {mean_rate:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
    if p_value > 0.05:
        print("PASS: Simulation matches theoretical prediction")
    else:
        print("FAIL: Simulation differs from theory")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(detection_rates, bins=15, alpha=0.7, edgecolor="black", color="#3498db")
    ax.axvline(
        p_theoretical,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Theoretical p = {p_theoretical}",
    )
    ax.axvline(
        mean_rate,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Measured mean = {mean_rate:.3f}",
    )
    ax.set_xlabel("Detection Rate")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Simulated Detection Rates")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()


def validate_performance_degradation():
    """Validation 3: Performance Degradation (Section 6.3) - Degradation table: T₀=100, α"""
    # Degradation table: T₀=100 Mbps, α=0.5 → steep decay. High load to induce throughput decay.
    config = {
        "network": {"bandwidth": 100, "buffer_size": 500, "latency": 10},
        "ids": {"detection_prob": 0.70},  # Binomial p=0.70
        "attacks": {"rate": 2.0, "packets_per_attack": 100},  # Heavy load
        "simulation": {"duration_minutes": 10, "sampling_interval": 0.5},
    }

    # Run simulation
    sim = NetworkAttackSimulation(config)
    sim.run(duration_minutes=10, seed=42)
    results = sim.metrics.get_dataframe()

    time = results["time"].values
    throughput = results["throughput_mbps"].values

    # Fit exponential: T(t) = T0 * exp(-alpha * t)
    def exponential_decay(t, T0, alpha):
        return T0 * np.exp(-alpha * t)

    popt, _ = curve_fit(exponential_decay, time, throughput, p0=[throughput[0], 0.01])
    T0_fit, alpha_fit = popt

    # Calculate R²
    residuals = throughput - exponential_decay(time, T0_fit, alpha_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((throughput - np.mean(throughput)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print("Exponential Fit Results:")
    print(f" T0 = {T0_fit:.2f} Mbps")
    print(f" alpha = {alpha_fit:.4f} per second")
    print(f" R² = {r_squared:.4f}")
    if r_squared > 0.90:
        print("PASS: Exponential model fits well")
    else:
        print("WARNING: Poor fit - consider alternative models")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(time, throughput, alpha=0.5, s=10, label="Simulation")
    ax.plot(
        time,
        exponential_decay(time, T0_fit, alpha_fit),
        "r-",
        linewidth=2,
        label=f"Fit: R²={r_squared:.3f}",
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Throughput Degradation with Exponential Fit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Example: run all validations
    validate_attack_generation()
    validate_ids_detection()
    validate_performance_degradation()

