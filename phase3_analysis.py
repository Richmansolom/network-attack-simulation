"""
Phase 3 Analysis Pipeline
Experimentation and Statistical Analysis

This script follows the Phase 3 guide:
- Section 0: Synthetic data generator
- Section 1: Experiment matrix builder
- Section 2: Experiment runner
- Sections 3-9: H1-H5 analyses, effect sizes, Bonferroni, confidence intervals
"""

import itertools
import json
import logging
import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

from src.simulator import NetworkAttackSimulation

plt.style.use("seaborn-v0_8-whitegrid")


def default_detection_prob_levels(
    lo: float = 0.3, hi: float = 0.95, step: float = 0.05
) -> list:
    """Arithmetic p grid for H1 supplement (default lo=0.30 matches abstract; not 0.10-0.95)."""
    lo, hi, step = float(lo), float(hi), float(step)
    if not (0.0 < lo < hi < 1.0) or step <= 0:
        raise ValueError("default_detection_prob_levels requires 0 < lo < hi < 1 and step > 0")
    n = int(np.round((hi - lo) / step)) + 1
    vals = lo + step * np.arange(n, dtype=float)
    vals = np.clip(vals, lo, hi)
    return sorted(np.unique(np.round(vals, 4)).tolist())


@dataclass
class Phase3Config:
    detection_probs: list = None
    attack_rates: list = None
    decay_rates: list = None
    packets_per_attack: list = None
    n_replications: int = 30
    simulation_duration_min: float = 10.0
    base_seed: int = 42
    experiment_plan_csv: str = "experiment_plan.csv"
    results_dir: str = "results"
    analysis_dir: str = "analysis"
    # Main factorial uses guide p levels when detection_probs is None (see __post_init__).
    # H1-only: extra runs for p values in h1_supplement_detection_probs that are not in
    # detection_probs (default grid 0.30-0.95 step 0.05, aligned with abstract); merged
    # only for analyze_h1 / H1 effect sizes / forest plot.
    run_h1_supplement: bool = True
    h1_supplement_detection_probs: list = None
    h1_supplement_experiment_plan_csv: str = "h1_supplement_plan.csv"
    h1_supplement_results_csv: str = "h1_supplement_results.csv"
    h1_supplement_lambda: float = 1.0
    h1_supplement_alpha: float = 0.3
    # === MODIFY THESE VALUES ===
    # Execution switches (set these manually per phase step)
    run_synthetic: bool = False
    run_experiments: bool = True
    run_analysis: bool = True
    show_figures: bool = True
    # ===========================

    def __post_init__(self):
        if self.detection_probs is None:
            self.detection_probs = [0.70, 0.80, 0.85, 0.90, 0.95]
        if self.h1_supplement_detection_probs is None:
            self.h1_supplement_detection_probs = default_detection_prob_levels(0.3, 0.95, 0.05)
        if self.attack_rates is None:
            self.attack_rates = [0.2, 0.5, 1.0, 2.0]
        if self.decay_rates is None:
            self.decay_rates = [0.1, 0.3, 0.5]
        if self.packets_per_attack is None:
            self.packets_per_attack = [10, 25, 50, 100]


def ensure_dirs(cfg: Phase3Config):
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.analysis_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(cfg.analysis_dir, "tables"), exist_ok=True)


# ===========================
# Section 0: Synthetic Data
# ===========================
def generate_synthetic_data(
    detection_probs=None,
    attack_rates=None,
    decay_rates=None,
    packets_list=None,
    n_reps=5,
    duration=10.0,
    base_seed=42,
    out_csv="synthetic_results.csv",
):
    if detection_probs is None:
        detection_probs = [0.70, 0.85, 0.95]
    if attack_rates is None:
        attack_rates = [0.5, 1.0, 2.0]
    if decay_rates is None:
        decay_rates = [0.1, 0.3, 0.5]
    if packets_list is None:
        packets_list = [10, 50]

    np.random.seed(base_seed)
    rows = []
    run_id = 0
    conditions = list(
        itertools.product(detection_probs, attack_rates, decay_rates, packets_list)
    )
    for cond_id, (p, lam, alpha, n_pkt) in enumerate(conditions):
        for rep in range(n_reps):
            seed = base_seed + run_id
            rng = np.random.default_rng(seed)

            total_attacks = int(rng.poisson(lam * duration))
            if total_attacks == 0:
                total_attacks = 1

            total_detected = 0
            for _ in range(total_attacks):
                packets_detected = rng.binomial(n_pkt, p)
                if packets_detected >= 1:
                    total_detected += 1

            observed_detection_rate = total_detected / total_attacks
            observed_packet_detection_rate = float(np.clip(p + rng.normal(0, 0.02), 0, 1))

            counts_per_min = [int(rng.poisson(lam)) for _ in range(int(duration))]
            n_arrivals = sum(counts_per_min)
            if n_arrivals > 1:
                inter_arrivals = rng.exponential(1.0 / lam, size=max(n_arrivals - 1, 1)).tolist()
            else:
                inter_arrivals = [float(rng.exponential(1.0 / lam))]

            T0 = 100.0
            times = np.linspace(0, duration, 20)
            throughput_series = T0 * np.exp(-alpha * times)
            throughput_series += rng.normal(0, 1.0, size=len(times))
            throughput_series = np.clip(throughput_series, 0, T0)

            rows.append(
                {
                    "run_id": run_id,
                    "condition_id": cond_id,
                    "replication": rep,
                    "p_detection": p,
                    "lambda_attack_rate": lam,
                    "alpha_decay_rate": alpha,
                    "n_packets_per_attack": n_pkt,
                    "sim_duration_min": duration,
                    "random_seed": seed,
                    "observed_detection_rate": observed_detection_rate,
                    "observed_packet_detection_rate": observed_packet_detection_rate,
                    "total_attacks": total_attacks,
                    "total_detected": total_detected,
                    "mean_inter_arrival_time": float(np.mean(inter_arrivals)),
                    "attack_counts_per_interval": json.dumps(counts_per_min),
                    "inter_arrival_times": json.dumps([round(x, 4) for x in inter_arrivals]),
                    "final_throughput": float(throughput_series[-1]),
                    "throughput_timeseries": json.dumps(
                        [[round(t, 2), round(v, 2)] for t, v in zip(times, throughput_series)]
                    ),
                    "mean_throughput": float(np.mean(throughput_series)),
                    "initial_throughput": T0,
                }
            )
            run_id += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Generated {len(df)} synthetic runs across {len(conditions)} conditions")
    return df


# ==============================
# Section 1: Experiment Matrix
# ==============================
def build_experiment_plan(cfg: Phase3Config) -> pd.DataFrame:
    print(f"Detection probability levels (p): {cfg.detection_probs}")
    conditions = list(
        itertools.product(
            cfg.detection_probs,
            cfg.attack_rates,
            cfg.decay_rates,
            cfg.packets_per_attack,
        )
    )

    experiments = []
    run_id = 0
    for cond_id, (p, lam, alpha, n) in enumerate(conditions):
        for rep in range(cfg.n_replications):
            experiments.append(
                {
                    "run_id": run_id,
                    "condition_id": cond_id,
                    "replication": rep,
                    "p_detection": p,
                    "lambda_attack_rate": lam,
                    "alpha_decay_rate": alpha,
                    "n_packets_per_attack": n,
                    "sim_duration_min": cfg.simulation_duration_min,
                    "random_seed": cfg.base_seed + run_id,
                }
            )
            run_id += 1

    plan = pd.DataFrame(experiments)
    plan.to_csv(cfg.experiment_plan_csv, index=False)
    print(f"Number of conditions: {len(conditions)}")
    print(f"Replications per condition: {cfg.n_replications}")
    print(f"Total simulation runs: {len(plan)}")
    print(f"Experiment plan saved: {cfg.experiment_plan_csv}")
    return plan


def build_h1_supplement_plan(cfg: Phase3Config) -> pd.DataFrame:
    """
    Extra factorial slice for H1 only: each p not in the main guide grid, crossed with
    all packets_per_attack, at fixed lambda and alpha.
    """
    main_ps = {round(float(x), 6) for x in cfg.detection_probs}
    extra_ps = sorted(
        {
            round(float(p), 4)
            for p in cfg.h1_supplement_detection_probs
            if round(float(p), 6) not in main_ps
        }
    )
    if not extra_ps:
        print(
            "H1 supplement: no p levels outside main factorial; skipping H1 supplement plan."
        )
        return pd.DataFrame()

    conditions = list(
        itertools.product(
            extra_ps,
            [float(cfg.h1_supplement_lambda)],
            [float(cfg.h1_supplement_alpha)],
            cfg.packets_per_attack,
        )
    )
    experiments = []
    run_id = 0
    seed_offset = 1_000_000
    for cond_id, (p, lam, alpha, n) in enumerate(conditions):
        for rep in range(cfg.n_replications):
            experiments.append(
                {
                    "run_id": run_id,
                    "condition_id": 10_000 + cond_id,
                    "replication": rep,
                    "p_detection": p,
                    "lambda_attack_rate": lam,
                    "alpha_decay_rate": alpha,
                    "n_packets_per_attack": n,
                    "sim_duration_min": cfg.simulation_duration_min,
                    "random_seed": int(cfg.base_seed) + seed_offset + run_id,
                }
            )
            run_id += 1

    plan = pd.DataFrame(experiments)
    plan.to_csv(cfg.h1_supplement_experiment_plan_csv, index=False)
    print(
        f"H1 supplement: {len(extra_ps)} extra p levels x {len(cfg.packets_per_attack)} n "
        f"x {cfg.n_replications} reps -> {len(plan)} runs"
    )
    print(f"  p values: {extra_ps}")
    print(
        f"  fixed lambda={cfg.h1_supplement_lambda}, alpha={cfg.h1_supplement_alpha}"
    )
    print(f"  plan saved: {cfg.h1_supplement_experiment_plan_csv}")
    return plan


def run_h1_supplement_experiments(cfg: Phase3Config) -> None:
    if not cfg.run_h1_supplement:
        return
    plan = build_h1_supplement_plan(cfg)
    if len(plan) == 0:
        return
    log_name = "experiment_log_h1_supplement.txt"
    runner = ExperimentRunner(
        cfg.h1_supplement_experiment_plan_csv,
        output_dir=cfg.results_dir,
        results_filename=cfg.h1_supplement_results_csv,
        log_file=log_name,
    )
    runner.run_all(save_every=50)


# ============================
# Section 2: Experiment Runner
# ============================
class ExperimentRunner:
    def __init__(
        self,
        experiment_plan_csv,
        output_dir="results",
        results_filename="all_results.csv",
        log_file=None,
    ):
        self.plan = pd.read_csv(experiment_plan_csv)
        self.output_dir = output_dir
        self.results_file = os.path.join(output_dir, results_filename)
        os.makedirs(output_dir, exist_ok=True)

        log_name = log_file or "experiment_log.txt"
        logging.basicConfig(
            filename=os.path.join(output_dir, log_name),
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )

        if os.path.exists(self.results_file):
            self.existing = pd.read_csv(self.results_file)
            self.completed_ids = set(self.existing["run_id"].values)
            print(f"Resuming: {len(self.completed_ids)} runs already completed")
        else:
            self.existing = pd.DataFrame()
            self.completed_ids = set()

    def run_single(self, row):
        """
        Run one simulation with the given parameters.
        Connected to your Phase 2 simulation implementation.
        """
        config = {
            "ids": {"detection_prob": float(row["p_detection"])},
            "attacks": {
                "rate": float(row["lambda_attack_rate"]),
                "packets_per_attack": int(row["n_packets_per_attack"]),
            },
            "network": {
                "bandwidth": 100,
                "buffer_size": 1000,
                "latency": 10,
                "degradation_alpha": float(row["alpha_decay_rate"]),
            },
            "simulation": {
                "duration_minutes": float(row["sim_duration_min"]),
                "sampling_interval": 1.0,
            },
        }

        sim = NetworkAttackSimulation(config)
        sim.run(duration_minutes=float(row["sim_duration_min"]), seed=int(row["random_seed"]))
        summary = sim.get_phase3_summary()

        ts_minutes = [[round(t / 60.0, 4), round(v, 6)] for t, v in summary["throughput_timeseries"]]

        return {
            "observed_detection_rate": float(summary["observed_detection_rate"]),
            "observed_packet_detection_rate": float(summary["observed_packet_detection_rate"]),
            "total_attacks": int(summary["total_attacks"]),
            "total_detected": int(summary["total_detected"]),
            "mean_inter_arrival_time": float(summary["mean_inter_arrival_time"]),
            "attack_counts_per_interval": json.dumps(summary["attack_counts_per_interval"]),
            "inter_arrival_times": json.dumps(
                [round(x, 6) for x in summary["inter_arrival_times"]]
            ),
            "final_throughput": float(summary["final_throughput"]),
            "throughput_timeseries": json.dumps(ts_minutes),
            "mean_throughput": float(summary["mean_throughput"]),
            "initial_throughput": float(summary["initial_throughput"]),
        }

    def run_all(self, save_every=50):
        remaining = self.plan[~self.plan["run_id"].isin(self.completed_ids)]
        total = len(remaining)
        print(f"Running {total} experiments ({len(self.completed_ids)} already done)")
        results = []
        start_time = time.time()

        for i, (_, row) in enumerate(remaining.iterrows()):
            try:
                result = self.run_single(row)
                combined = {**row.to_dict(), **result}
                results.append(combined)

                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    remaining_time = (total - i - 1) / rate if rate > 0 else 0
                    print(
                        f"Run {i+1}/{total} | {rate:.2f} runs/sec | "
                        f"~{remaining_time/60:.1f} min remaining"
                    )

                if (i + 1) % save_every == 0:
                    self._save_results(results)
                    results = []

                logging.info(f"Run {int(row['run_id'])} completed successfully")
            except Exception as exc:
                logging.error(f"Run {int(row['run_id'])} FAILED: {str(exc)}")
                print(f"ERROR on run {int(row['run_id'])}: {exc}")

        if results:
            self._save_results(results)

        elapsed_total = time.time() - start_time
        print(f"Complete! {total} runs in {elapsed_total/60:.1f} minutes")

    def _save_results(self, new_results):
        new_df = pd.DataFrame(new_results)
        if os.path.exists(self.results_file):
            existing = pd.read_csv(self.results_file)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(self.results_file, index=False)
        self.completed_ids.update(new_df["run_id"].values)
        print(f"Saved. Total completed: {len(self.completed_ids)}")


# ============================
# Sections 3-9: Analyses
# ============================
def analyze_h1(df: pd.DataFrame, out_tables: str, out_figs: str, show_figures: bool = True):
    print("=" * 65)
    print("HYPOTHESIS 1: BINOMIAL DETECTION MODEL")
    print("=" * 65)
    results_h1 = []
    for p in sorted(df["p_detection"].unique()):
        for n in sorted(df["n_packets_per_attack"].unique()):
            subset = df[(df["p_detection"] == p) & (df["n_packets_per_attack"] == n)]
            # H1 metric is undefined when total_attacks == 0 (0/0). Exclude those runs.
            subset = subset[subset["total_attacks"] > 0]
            if len(subset) == 0:
                continue
            p_theory = 1 - (1 - p) ** n
            observed = subset["observed_detection_rate"].values
            obs_mean = np.mean(observed)
            obs_std = np.std(observed, ddof=1) if len(observed) > 1 else 0.0
            n_unique = int(np.unique(observed).size)
            degenerate = bool(obs_std == 0)
            test_used = "one_sample_t_test"
            if obs_std == 0:
                # Degenerate case: run-level rates are constant.
                # Use exact binomial test over pooled attack outcomes.
                test_used = "exact_binomial_fallback"
                t_stat = np.nan
                total_successes = int(subset["total_detected"].sum())
                total_trials = int(subset["total_attacks"].sum())
                if total_trials > 0:
                    p_value = float(
                        stats.binomtest(total_successes, total_trials, p_theory, alternative="two-sided").pvalue
                    )
                else:
                    p_value = np.nan
            else:
                t_stat, p_value = stats.ttest_1samp(observed, p_theory)
            se = obs_std / np.sqrt(len(observed)) if len(observed) > 0 else 0.0
            ci_lower = obs_mean - 1.96 * se
            ci_upper = obs_mean + 1.96 * se
            theory_in_ci = ci_lower <= p_theory <= ci_upper
            results_h1.append(
                {
                    "p": p,
                    "n": n,
                    "n_runs": int(len(observed)),
                    "n_unique_values": n_unique,
                    "degenerate": degenerate,
                    "test_used": test_used,
                    "theoretical": p_theory,
                    "observed_mean": obs_mean,
                    "observed_std": obs_std,
                    "difference": obs_mean - p_theory,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "theory_in_ci": theory_in_ci,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                }
            )
            status = "MATCH" if np.isfinite(p_value) and p_value >= 0.05 else "DIFFERS"
            p_display = f"{p_value:.4f}" if np.isfinite(p_value) else "nan"
            n_int = int(n)
            print(
                f"p={p:.2f}, n={n_int:3d}: theory={p_theory:.8f}, "
                f"observed={obs_mean:.4f} +/- {obs_std:.4f}, "
                f"p-value={p_display}, test={test_used} [{status}]"
            )

    h1_table = pd.DataFrame(results_h1)
    h1_table.to_csv(os.path.join(out_tables, "h1_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    n_range = np.arange(1, 105)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    n_levels = df["n_packets_per_attack"].unique()

    for i, p in enumerate(sorted(df["p_detection"].unique())):
        theory_curve = 1 - (1 - p) ** n_range
        ax.plot(
            n_range,
            theory_curve,
            "-",
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"Theory p={p:.2f}",
        )
        for n in n_levels:
            subset = df[(df["p_detection"] == p) & (df["n_packets_per_attack"] == n)]
            if len(subset) == 0:
                continue
            obs_mean = subset["observed_detection_rate"].mean()
            obs_se = subset["observed_detection_rate"].std(ddof=1) / np.sqrt(len(subset))
            yerr = 1.96 * obs_se if np.isfinite(obs_se) else 0
            ax.errorbar(
                n,
                obs_mean,
                yerr=yerr,
                fmt="o",
                color=colors[i % len(colors)],
                markersize=7,
                capsize=4,
                capthick=1.2,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )

    ax.set_xlabel("Packets per Attack (n)", fontsize=13)
    ax.set_ylabel("P(Detect at Least 1 Packet)", fontsize=13)
    ax.set_title(
        "Hypothesis 1: Theoretical vs. Simulated Detection Probability",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(out_figs, "h1_detection_overlay.png"), dpi=300, bbox_inches="tight")
    if show_figures:
        plt.show()
    else:
        plt.close(fig)
    print("Figure saved: h1_detection_overlay.png")
    return h1_table


def analyze_h2(df: pd.DataFrame, out_tables: str, out_figs: str, show_figures: bool = True):
    print("=" * 65)
    print("HYPOTHESIS 2: POISSON ATTACK COUNTS")
    print("=" * 65)
    results_h2 = []
    lambdas = sorted(df["lambda_attack_rate"].unique())
    for lam in lambdas:
        subset = df[df["lambda_attack_rate"] == lam]
        duration = float(subset["sim_duration_min"].iloc[0])
        expected_mean = lam * duration
        attack_counts = subset["total_attacks"].values
        max_count = max(attack_counts.max(), int(expected_mean + 4 * np.sqrt(expected_mean)))
        bins = np.arange(0, max_count + 2)
        observed_freq, _ = np.histogram(attack_counts, bins=bins)
        expected_freq = np.array(
            [stats.poisson.pmf(k, expected_mean) * len(attack_counts) for k in range(len(observed_freq))]
        )
        obs_combined = []
        exp_combined = []
        obs_accum = 0
        exp_accum = 0
        for o, e in zip(observed_freq, expected_freq):
            obs_accum += o
            exp_accum += e
            if exp_accum >= 5:
                obs_combined.append(obs_accum)
                exp_combined.append(exp_accum)
                obs_accum = 0
                exp_accum = 0
        if obs_accum > 0 or exp_accum > 0:
            if len(obs_combined) > 0:
                obs_combined[-1] += obs_accum
                exp_combined[-1] += exp_accum
            else:
                obs_combined.append(obs_accum)
                exp_combined.append(exp_accum)
        # Align totals before chi-square (required by scipy)
        obs_total = float(np.sum(obs_combined))
        exp_total = float(np.sum(exp_combined))
        if exp_total > 0:
            scale = obs_total / exp_total
            exp_combined = [e * scale for e in exp_combined]
        chi2, p_value = stats.chisquare(obs_combined, exp_combined)
        results_h2.append(
            {
                "lambda": lam,
                "expected_mean": expected_mean,
                "observed_mean": float(np.mean(attack_counts)),
                "observed_std": float(np.std(attack_counts, ddof=1)),
                "chi2_statistic": chi2,
                "p_value": p_value,
                "n_bins": len(obs_combined),
            }
        )
        status = "MATCHES POISSON" if p_value > 0.05 else "DEVIATES"
        print(
            f"lambda={lam:.1f}: E[N]={expected_mean:.1f}, "
            f"observed mean={np.mean(attack_counts):.2f}, "
            f"chi2={chi2:.2f}, p={p_value:.4f} [{status}]"
        )

    h2_table = pd.DataFrame(results_h2)
    h2_table.to_csv(os.path.join(out_tables, "h2_results.csv"), index=False)

    fig, axes = plt.subplots(
        1,
        len(lambdas),
        figsize=(6 * len(lambdas), 5.5),
        sharey=False,
        constrained_layout=True,
    )
    if len(lambdas) == 1:
        axes = [axes]
    for ax, lam in zip(axes, lambdas):
        subset = df[df["lambda_attack_rate"] == lam]
        duration = float(subset["sim_duration_min"].iloc[0])
        expected_mean = lam * duration
        counts = subset["total_attacks"].values
        max_k = int(max(counts.max(), expected_mean + 4 * np.sqrt(expected_mean)))
        bins = np.arange(-0.5, max_k + 1.5)
        ax.hist(
            counts,
            bins=bins,
            density=True,
            alpha=0.75,
            edgecolor="black",
            label="Simulated",
            color="steelblue",
        )
        k_vals = np.arange(0, max_k + 1)
        pmf_vals = stats.poisson.pmf(k_vals, expected_mean)
        ax.plot(k_vals, pmf_vals, "ro-", markersize=6, linewidth=2, label=f"Poisson({expected_mean:.0f})")
        ax.set_xlabel("Number of Attacks", fontsize=11)
        ax.set_ylabel("Probability", fontsize=11)
        ax.set_title(f"lambda = {lam}", fontsize=12, fontweight="bold")
        ax.set_xticks(np.arange(0, max_k + 1, max(1, int(max_k / 10))))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Hypothesis 2: Observed vs. Poisson Attack Counts", fontsize=15, fontweight="bold")
    plt.savefig(os.path.join(out_figs, "h2_poisson_overlay.png"), dpi=300, bbox_inches="tight")
    if show_figures:
        plt.show()
    plt.close(fig)
    return h2_table


def analyze_h3(df: pd.DataFrame, out_tables: str, out_figs: str, show_figures: bool = True):
    print("=" * 65)
    print("HYPOTHESIS 3: EXPONENTIAL INTER-ARRIVAL TIMES")
    print("=" * 65)
    results_h3 = []
    lambdas = sorted(df["lambda_attack_rate"].unique())
    for lam in lambdas:
        subset = df[df["lambda_attack_rate"] == lam]
        all_inter_arrivals = []
        for _, row in subset.iterrows():
            try:
                times = json.loads(row["inter_arrival_times"])
                all_inter_arrivals.extend(times)
            except (json.JSONDecodeError, TypeError):
                continue
        if len(all_inter_arrivals) < 10:
            print(f"lambda={lam}: Too few inter-arrival times ({len(all_inter_arrivals)})")
            continue
        ia = np.array(all_inter_arrivals, dtype=float)
        ks_stat, p_value = stats.kstest(ia, "expon", args=(0, 1.0 / lam))
        results_h3.append(
            {
                "lambda": lam,
                "theoretical_mean": 1.0 / lam,
                "observed_mean": float(np.mean(ia)),
                "observed_median": float(np.median(ia)),
                "theoretical_median": float(np.log(2) / lam),
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "n_samples": len(ia),
            }
        )
        status = "MATCHES EXPONENTIAL" if p_value > 0.05 else "DEVIATES"
        print(
            f"lambda={lam:.1f}: E[T]={1.0/lam:.2f} min, observed mean={np.mean(ia):.2f}, "
            f"KS={ks_stat:.4f}, p={p_value:.4f} [{status}]"
        )

    h3_table = pd.DataFrame(results_h3)
    h3_table.to_csv(os.path.join(out_tables, "h3_results.csv"), index=False)

    fig, axes = plt.subplots(
        2,
        len(lambdas),
        figsize=(6 * len(lambdas), 9),
        constrained_layout=True,
    )
    if len(lambdas) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for col, lam in enumerate(lambdas):
        subset = df[df["lambda_attack_rate"] == lam]
        all_ia = []
        for _, row in subset.iterrows():
            try:
                all_ia.extend(json.loads(row["inter_arrival_times"]))
            except Exception:
                continue
        ia = np.array(all_ia, dtype=float)
        if len(ia) == 0:
            continue
        ax_hist = axes[0, col]
        ax_hist.hist(
            ia,
            bins=40,
            density=True,
            alpha=0.75,
            edgecolor="black",
            color="steelblue",
            label="Simulated",
        )
        t_range = np.linspace(0, np.percentile(ia, 99), 200)
        pdf_theory = lam * np.exp(-lam * t_range)
        ax_hist.plot(t_range, pdf_theory, "r-", linewidth=2.5, label=f"Exp(lambda={lam})")
        ax_hist.set_xlabel("Inter-Arrival Time (min)")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title(f"lambda = {lam}", fontweight="bold")
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, alpha=0.3)

        ax_qq = axes[1, col]
        q = np.linspace(0.01, 0.99, min(len(ia), 200))
        theoretical_quantiles = stats.expon.ppf(q, scale=1.0 / lam)
        sample_quantiles = np.quantile(ia, q)
        ax_qq.scatter(theoretical_quantiles, sample_quantiles, alpha=0.5, s=10)
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        ax_qq.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect fit")
        ax_qq.set_xlabel("Theoretical Quantiles")
        ax_qq.set_ylabel("Sample Quantiles")
        ax_qq.set_title(f"Q-Q Plot: lambda = {lam}", fontweight="bold")
        ax_qq.legend()
        ax_qq.grid(True, alpha=0.3)
    fig.suptitle("Hypothesis 3: Exponential Inter-Arrival Time Analysis", fontsize=15, fontweight="bold")
    plt.savefig(os.path.join(out_figs, "h3_exponential_analysis.png"), dpi=300, bbox_inches="tight")
    if show_figures:
        plt.show()
    plt.close(fig)
    return h3_table


def decay_model(t, T0, alpha):
    return T0 * np.exp(-alpha * t)


def analyze_h4(df: pd.DataFrame, out_tables: str, out_figs: str, show_figures: bool = True):
    print("=" * 65)
    print("HYPOTHESIS 4: THROUGHPUT DEGRADATION (EXPONENTIAL DECAY)")
    print("=" * 65)
    results_h4 = []
    alphas = sorted(df["alpha_decay_rate"].unique())
    for alpha_theory in alphas:
        subset = df[df["alpha_decay_rate"] == alpha_theory]
        fitted_alphas = []
        r_squared_values = []
        for _, row in subset.iterrows():
            try:
                ts = json.loads(row["throughput_timeseries"])
                times = np.array([pt[0] for pt in ts], dtype=float)
                throughputs = np.array([pt[1] for pt in ts], dtype=float)
            except (json.JSONDecodeError, TypeError, IndexError):
                continue
            if len(times) < 5:
                continue
            try:
                popt, _ = curve_fit(
                    decay_model,
                    times,
                    throughputs,
                    p0=[100.0, alpha_theory],
                    bounds=([0, 0], [200, 10]),
                    maxfev=5000,
                )
                fitted_T0, fitted_alpha = popt
                predicted = decay_model(times, *popt)
                ss_res = np.sum((throughputs - predicted) ** 2)
                ss_tot = np.sum((throughputs - np.mean(throughputs)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                fitted_alphas.append(float(fitted_alpha))
                r_squared_values.append(float(r2))
            except RuntimeError:
                continue
        if len(fitted_alphas) == 0:
            print(f"alpha={alpha_theory:.1f}: No successful fits")
            continue
        fitted_alphas = np.array(fitted_alphas)
        mean_fitted = float(np.mean(fitted_alphas))
        std_fitted = float(np.std(fitted_alphas, ddof=1)) if len(fitted_alphas) > 1 else 0.0
        mean_r2 = float(np.mean(r_squared_values))
        bias = mean_fitted - alpha_theory
        results_h4.append(
            {
                "alpha_theoretical": alpha_theory,
                "alpha_fitted_mean": mean_fitted,
                "alpha_fitted_std": std_fitted,
                "bias": bias,
                "bias_pct": 100 * bias / alpha_theory if alpha_theory != 0 else 0,
                "mean_r_squared": mean_r2,
                "n_fits": len(fitted_alphas),
                "half_life_theory": float(np.log(2) / alpha_theory),
                "half_life_fitted": float(np.log(2) / mean_fitted) if mean_fitted > 0 else np.nan,
            }
        )
        print(
            f"alpha={alpha_theory:.1f}: fitted={mean_fitted:.4f} +/- {std_fitted:.4f}, "
            f"bias={bias:+.4f} ({100*bias/alpha_theory:+.1f}%), R2={mean_r2:.4f}"
        )
    h4_table = pd.DataFrame(results_h4)
    h4_table.to_csv(os.path.join(out_tables, "h4_results.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    ax1 = axes[0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, alpha in enumerate(alphas):
        subset = df[df["alpha_decay_rate"] == alpha]
        for _, row in subset.head(3).iterrows():
            try:
                ts = json.loads(row["throughput_timeseries"])
                times = [pt[0] for pt in ts]
                throughputs = [pt[1] for pt in ts]
                ax1.plot(
                    times,
                    throughputs,
                    "-",
                    color=colors[i % len(colors)],
                    alpha=0.2,
                    linewidth=0.8,
                )
            except Exception:
                continue
        t_theory = np.linspace(0, 10, 100)
        T_theory = 100 * np.exp(-alpha * t_theory)
        ax1.plot(t_theory, T_theory, "--", color=colors[i % len(colors)], linewidth=2.5, label=f"Theory alpha={alpha}")
    ax1.set_xlabel("Time (minutes)", fontsize=12)
    ax1.set_ylabel("Throughput (Mbps)", fontsize=12)
    ax1.set_title("Throughput Degradation: Theory vs. Simulation", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    if len(h4_table) > 0:
        ax2.errorbar(
            h4_table["alpha_theoretical"],
            h4_table["alpha_fitted_mean"],
            yerr=h4_table["alpha_fitted_std"],
            fmt="o",
            markersize=10,
            capsize=5,
            capthick=2,
            color="steelblue",
            markeredgecolor="black",
        )
        max_alpha = max(h4_table["alpha_theoretical"].max(), h4_table["alpha_fitted_mean"].max()) * 1.1
        ax2.plot([0, max_alpha], [0, max_alpha], "r--", linewidth=2, label="Perfect agreement")
    ax2.set_xlabel("Theoretical alpha", fontsize=12)
    ax2.set_ylabel("Fitted alpha", fontsize=12)
    ax2.set_title("Fitted vs. Theoretical Decay Rate", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_figs, "h4_degradation_analysis.png"), dpi=300, bbox_inches="tight")
    if show_figures:
        plt.show()
    plt.close(fig)
    return h4_table


def analyze_h5(df: pd.DataFrame, out_tables: str, out_figs: str, show_figures: bool = True):
    print("=" * 65)
    print("HYPOTHESIS 5: MULTI-MODEL CROSS-VALIDATION")
    print("=" * 65)
    df = df.copy()
    # Detection-rate error is undefined when no attacks occurred (0/0 at run level).
    # Exclude those runs to avoid biasing interaction effects at low lambda.
    df = df[df["total_attacks"] > 0].copy()
    if df.empty:
        print("No valid runs with total_attacks > 0 for H5.")
        return df
    df["theoretical_detection"] = 1 - (1 - df["p_detection"]) ** df["n_packets_per_attack"]
    df["detection_error"] = df["observed_detection_rate"] - df["theoretical_detection"]
    df["detection_error_pct"] = 100 * df["detection_error"] / df["theoretical_detection"]

    interaction = df.groupby(["p_detection", "lambda_attack_rate"])["detection_error"].agg(["mean", "std"]).round(4)
    print("\nMean detection error by attack rate and IDS quality:")
    print(interaction.to_string())
    interaction.to_csv(os.path.join(out_tables, "h5_interaction_table.csv"))

    # ANOVA summary to align with guide's "ANOVA + interaction plots" wording.
    anova_rows = []
    for p in sorted(df["p_detection"].unique()):
        sub = df[df["p_detection"] == p]
        groups = [g["detection_error"].values for _, g in sub.groupby("lambda_attack_rate")]
        if len(groups) >= 2 and all(len(g) > 1 for g in groups):
            f_stat, p_val = stats.f_oneway(*groups)
            anova_rows.append(
                {
                    "analysis": "lambda_effect_within_p",
                    "p_detection": p,
                    "f_statistic": float(f_stat),
                    "p_value": float(p_val),
                }
            )
    for lam in sorted(df["lambda_attack_rate"].unique()):
        sub = df[df["lambda_attack_rate"] == lam]
        groups = [g["detection_error"].values for _, g in sub.groupby("p_detection")]
        if len(groups) >= 2 and all(len(g) > 1 for g in groups):
            f_stat, p_val = stats.f_oneway(*groups)
            anova_rows.append(
                {
                    "analysis": "p_effect_within_lambda",
                    "lambda_attack_rate": lam,
                    "f_statistic": float(f_stat),
                    "p_value": float(p_val),
                }
            )
    if anova_rows:
        anova_df = pd.DataFrame(anova_rows)
        anova_df.to_csv(os.path.join(out_tables, "h5_anova_summary.csv"), index=False)

    pivot = df.groupby(["p_detection", "lambda_attack_rate"])["detection_error"].mean().unstack()
    # Use the same rounded values for both color and labels so they never disagree.
    pivot_display = pivot.round(3)
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    abs_max = float(np.nanmax(np.abs(pivot_display.values)))
    all_zero_error = np.isclose(abs_max, 0.0, atol=1e-6)
    vmax = max(0.001, abs_max)
    cmap = "Blues" if all_zero_error else "RdBu_r"
    im = ax.imshow(pivot_display.values, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
    ax.set_xlabel("Attack Rate (lambda)", fontsize=12)
    ax.set_ylabel("Detection Probability (p)", fontsize=12)
    ax.set_title(
        "Hypothesis 5: Where Models Break Down\n"
        "(Prediction Error: Observed - Theoretical Detection Rate)",
        fontsize=13,
        fontweight="bold",
    )
    if all_zero_error:
        ax.text(
            0.5,
            1.06,
            "All cells are ~0.000 (no detectable interaction error).",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            color="dimgray",
        )
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot_display.values[i, j]
            color = "white" if abs(val) > 0.08 else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Prediction Error", shrink=0.8)
    plt.savefig(os.path.join(out_figs, "h5_error_heatmap.png"), dpi=300, bbox_inches="tight")
    if show_figures:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, p in enumerate(sorted(df["p_detection"].unique())):
        sub = df[df["p_detection"] == p]
        means = sub.groupby("lambda_attack_rate")["observed_detection_rate"].mean()
        stds = sub.groupby("lambda_attack_rate")["observed_detection_rate"].std()
        n_rep = sub.groupby("lambda_attack_rate")["observed_detection_rate"].count().values
        yerr = 1.96 * stds.values / np.sqrt(np.maximum(n_rep, 1))
        ax.errorbar(
            means.index,
            means.values,
            yerr=yerr,
            fmt="o-",
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=8,
            capsize=4,
            label=f"p = {p:.2f}",
        )
    ax.set_xlabel("Attack Rate (lambda)", fontsize=12)
    ax.set_ylabel("Mean Observed Detection Rate", fontsize=12)
    ax.set_title("Interaction Plot: IDS Quality x Attack Rate", fontsize=13, fontweight="bold")
    # When curves overlap at ~0/1 levels, tighten y-range for readability.
    y_all = []
    for p in sorted(df["p_detection"].unique()):
        sub = df[df["p_detection"] == p]
        y_all.extend(sub.groupby("lambda_attack_rate")["observed_detection_rate"].mean().values.tolist())
    if y_all and (max(y_all) - min(y_all) < 0.03):
        center = float(np.mean(y_all))
        ax.set_ylim(max(0.0, center - 0.03), min(1.0, center + 0.03))
    ax.legend(title="IDS Quality", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_figs, "h5_interaction_plot.png"), dpi=300, bbox_inches="tight")
    if show_figures:
        plt.show()
    plt.close(fig)
    return df


def effect_sizes_h1(df: pd.DataFrame, out_tables: str):
    def cohens_d_one_sample(data, theoretical_value):
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        return (np.mean(data) - theoretical_value) / std

    rows = []
    print("=" * 65)
    print("EFFECT SIZES (Cohen's d) FOR HYPOTHESIS 1")
    print("=" * 65)
    for p in sorted(df["p_detection"].unique()):
        for n in sorted(df["n_packets_per_attack"].unique()):
            subset = df[(df["p_detection"] == p) & (df["n_packets_per_attack"] == n)]
            if len(subset) == 0:
                continue
            theory = 1 - (1 - p) ** n
            observed = subset["observed_detection_rate"].values
            d = cohens_d_one_sample(observed, theory)
            label = (
                "negligible"
                if abs(d) < 0.2
                else "small"
                if abs(d) < 0.5
                else "medium"
                if abs(d) < 0.8
                else "LARGE"
            )
            print(f"p={p:.2f}, n={int(n):3d}: d = {d:+.3f} ({label})")
            rows.append({"p": p, "n": n, "cohens_d": d, "magnitude": label})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_tables, "h1_effect_sizes.csv"), index=False)
    return out


def bonferroni_summary(out_tables: str):
    n_hypotheses = 5
    alpha_original = 0.05
    alpha_corrected = alpha_original / n_hypotheses
    info = pd.DataFrame(
        [
            {"metric": "n_hypotheses", "value": n_hypotheses},
            {"metric": "alpha_original", "value": alpha_original},
            {"metric": "alpha_corrected", "value": alpha_corrected},
        ]
    )
    info.to_csv(os.path.join(out_tables, "multiple_comparisons.csv"), index=False)
    print(f"Bonferroni-corrected alpha: {alpha_corrected:.3f}")
    return alpha_corrected


def forest_plot_h1(df: pd.DataFrame, out_figs: str, show_figures: bool = True):
    fig, ax = plt.subplots(figsize=(11, 8.5), constrained_layout=True)
    labels = []
    y_positions = []
    pos = 0
    for p in sorted(df["p_detection"].unique(), reverse=True):
        for n in sorted(df["n_packets_per_attack"].unique()):
            subset = df[(df["p_detection"] == p) & (df["n_packets_per_attack"] == n)]
            if len(subset) == 0:
                continue
            theory = 1 - (1 - p) ** n
            obs = subset["observed_detection_rate"].values
            obs_mean = np.mean(obs)
            se = np.std(obs, ddof=1) / np.sqrt(len(obs)) if len(obs) > 1 else 0
            ci_low = obs_mean - 1.96 * se
            ci_high = obs_mean + 1.96 * se
            contains_theory = ci_low <= theory <= ci_high
            color = "#2ca02c" if contains_theory else "#d62728"
            ax.errorbar(
                obs_mean,
                pos,
                xerr=[[obs_mean - ci_low], [ci_high - obs_mean]],
                fmt="o",
                color=color,
                markersize=6,
                capsize=3,
                capthick=1.5,
                linewidth=1.5,
            )
            ax.plot(theory, pos, "|", color="black", markersize=12, markeredgewidth=2)
            labels.append(f"p={p:.2f}, n={n}")
            y_positions.append(pos)
            pos += 1
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Detection Rate", fontsize=12)
    ax.set_title(
        "H1: Observed vs. Theoretical Detection Rate\n"
        "(green = CI contains theory, red = CI excludes theory)",
        fontsize=13,
        fontweight="bold",
    )
    ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.3)
    plt.savefig(os.path.join(out_figs, "h1_forest_plot.png"), dpi=300, bbox_inches="tight")
    if show_figures:
        plt.show()
    plt.close(fig)


def run_quality_check(results_file: str):
    df = pd.read_csv(results_file)
    print("=== QUALITY CHECK ===")
    print(f"Total rows: {len(df)}")
    print(f"Unique conditions: {df['condition_id'].nunique()}")
    print(f"Replications per condition:\n{df.groupby('condition_id').size().describe()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    if len(df) > 0:
        sample = df[df["condition_id"] == df["condition_id"].iloc[0]]
        print(
            f"\nCondition {int(sample['condition_id'].iloc[0])} "
            f"(p={sample['p_detection'].iloc[0]}, lambda={sample['lambda_attack_rate'].iloc[0]}):"
        )
        print(
            f" Detection rate: {sample['observed_detection_rate'].mean():.3f} "
            f"+/- {sample['observed_detection_rate'].std():.3f}"
        )
        print(
            f" Packet detection rate: {sample['observed_packet_detection_rate'].mean():.3f} "
            f"+/- {sample['observed_packet_detection_rate'].std():.3f}"
        )
        print(
            f" Total attacks: {sample['total_attacks'].mean():.1f} "
            f"+/- {sample['total_attacks'].std():.1f}"
        )
        try:
            test_json = json.loads(sample["inter_arrival_times"].iloc[0])
            print(f" JSON parsing works: {len(test_json)} inter-arrival times")
        except Exception as exc:
            print(f" JSON PARSING FAILED: {exc}")


def run_full_analysis(cfg: Phase3Config):
    ensure_dirs(cfg)
    results_file = os.path.join(cfg.results_dir, "all_results.csv")
    if not os.path.exists(results_file):
        raise FileNotFoundError(
            f"{results_file} not found. Run experiments first or set run_synthetic=True."
        )
    df_main = pd.read_csv(results_file)
    supp_path = os.path.join(cfg.results_dir, cfg.h1_supplement_results_csv)
    if os.path.exists(supp_path):
        df_sup = pd.read_csv(supp_path)
        df_h1 = pd.concat([df_main, df_sup], ignore_index=True)
        print(
            f"H1 analysis: main factorial ({len(df_main)} runs) + "
            f"H1 supplement ({len(df_sup)} runs)."
        )
    else:
        df_h1 = df_main
        print("H1 supplement results not found; H1 uses main factorial only.")

    out_tables = os.path.join(cfg.analysis_dir, "tables")
    out_figs = os.path.join(cfg.analysis_dir, "figures")

    analyze_h1(df_h1, out_tables, out_figs, show_figures=cfg.show_figures)
    analyze_h2(df_main, out_tables, out_figs, show_figures=cfg.show_figures)
    analyze_h3(df_main, out_tables, out_figs, show_figures=cfg.show_figures)
    analyze_h4(df_main, out_tables, out_figs, show_figures=cfg.show_figures)
    df_h5 = analyze_h5(df_main, out_tables, out_figs, show_figures=cfg.show_figures)
    effect_sizes_h1(df_h1, out_tables)
    bonferroni_summary(out_tables)
    forest_plot_h1(df_h1, out_figs, show_figures=cfg.show_figures)
    print("Analysis complete. See analysis/figures and analysis/tables.")


def main():
    print("Phase 3: Experimentation and Statistical Analysis")
    print("Testing Theory Against Simulation")
    cfg = Phase3Config()
    ensure_dirs(cfg)

    if cfg.run_synthetic:
        print("\n[Part 1 Prep] Generating synthetic data...")
        generate_synthetic_data(out_csv=os.path.join(cfg.results_dir, "all_results.csv"))

    print("\n[Part 1] Building experiment matrix...")
    build_experiment_plan(cfg)

    if cfg.run_experiments:
        print("\n[Part 2] Running experiments...")
        runner = ExperimentRunner(cfg.experiment_plan_csv, output_dir=cfg.results_dir)
        runner.run_all(save_every=50)
        run_h1_supplement_experiments(cfg)
        run_quality_check(os.path.join(cfg.results_dir, "all_results.csv"))

    if cfg.run_analysis:
        print("\n[Part 3] Running statistical analysis...")
        run_full_analysis(cfg)


if __name__ == "__main__":
    main()
