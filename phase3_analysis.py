"""
Phase 3 Analysis Pipeline (Fresh Build)
======================================

Guide-aligned default design:
  5 p x 4 lambda x 3 alpha x 4 n = 240 conditions
  240 x 30 replications = 7,200 runs
"""

import itertools
import json
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


@dataclass
class Phase3Config:
    # Guide defaults
    detection_probs: list = None  # [0.70, 0.80, 0.85, 0.90, 0.95]
    attack_rates: list = None  # [0.2, 0.5, 1.0, 2.0]
    decay_rates: list = None  # [0.1, 0.3, 0.5]
    packets_per_attack: list = None  # [10, 25, 50, 100]
    n_replications: int = 30
    simulation_duration_min: float = 10.0
    base_seed: int = 42

    # IO
    experiment_plan_csv: str = "experiment_plan.csv"
    results_dir: str = "results"
    analysis_dir: str = "analysis"

    # Execution switches
    run_synthetic: bool = False
    run_experiments: bool = True
    run_analysis: bool = True
    show_figures: bool = True

    def __post_init__(self):
        if self.detection_probs is None:
            self.detection_probs = [0.70, 0.80, 0.85, 0.90, 0.95]
        if self.attack_rates is None:
            self.attack_rates = [0.2, 0.5, 1.0, 2.0]
        if self.decay_rates is None:
            self.decay_rates = [0.1, 0.3, 0.5]
        if self.packets_per_attack is None:
            self.packets_per_attack = [10, 25, 50, 100]


def ensure_dirs(cfg: Phase3Config):
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.analysis_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(cfg.analysis_dir, "figures"), exist_ok=True)


def generate_synthetic_data(cfg: Phase3Config, out_csv: str):
    rng = np.random.default_rng(cfg.base_seed)
    rows = []
    run_id = 0
    conditions = list(
        itertools.product(
            cfg.detection_probs,
            cfg.attack_rates,
            cfg.decay_rates,
            cfg.packets_per_attack,
        )
    )
    for cond_id, (p, lam, alpha, n_pkt) in enumerate(conditions):
        for rep in range(cfg.n_replications):
            seed = cfg.base_seed + run_id
            r = np.random.default_rng(seed)

            total_attacks = max(1, int(r.poisson(lam * cfg.simulation_duration_min)))
            total_detected = 0
            total_pkt_detected = 0
            for _ in range(total_attacks):
                pkt_detected = int(r.binomial(n_pkt, p))
                total_pkt_detected += pkt_detected
                if pkt_detected >= 1:
                    total_detected += 1

            obs_det = total_detected / total_attacks
            obs_pkt = total_pkt_detected / max(total_attacks * n_pkt, 1)

            counts_per_min = [int(r.poisson(lam)) for _ in range(int(cfg.simulation_duration_min))]
            n_arrivals = max(1, sum(counts_per_min))
            inter_arrivals = r.exponential(1.0 / lam, size=max(n_arrivals - 1, 1)).tolist()

            t = np.linspace(0, cfg.simulation_duration_min, 20)
            throughput = 100.0 * np.exp(-alpha * t) + r.normal(0, 1.2, size=len(t))
            throughput = np.clip(throughput, 0, 100.0)

            rows.append(
                {
                    "run_id": run_id,
                    "condition_id": cond_id,
                    "replication": rep,
                    "p_detection": p,
                    "lambda_attack_rate": lam,
                    "alpha_decay_rate": alpha,
                    "n_packets_per_attack": n_pkt,
                    "sim_duration_min": cfg.simulation_duration_min,
                    "random_seed": seed,
                    "observed_detection_rate": float(obs_det),
                    "observed_packet_detection_rate": float(obs_pkt),
                    "total_attacks": int(total_attacks),
                    "total_detected": int(total_detected),
                    "mean_inter_arrival_time": float(np.mean(inter_arrivals)),
                    "attack_counts_per_interval": json.dumps(counts_per_min),
                    "inter_arrival_times": json.dumps([round(x, 6) for x in inter_arrivals]),
                    "final_throughput": float(throughput[-1]),
                    "throughput_timeseries": json.dumps(
                        [[round(tt, 4), round(v, 6)] for tt, v in zip(t, throughput)]
                    ),
                    "mean_throughput": float(np.mean(throughput)),
                    "initial_throughput": 100.0,
                }
            )
            run_id += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Synthetic results written: {out_csv} ({len(df)} rows)")
    return df


def build_experiment_plan(cfg: Phase3Config):
    conditions = list(
        itertools.product(cfg.detection_probs, cfg.attack_rates, cfg.decay_rates, cfg.packets_per_attack)
    )
    rows = []
    run_id = 0
    for cond_id, (p, lam, alpha, n_pkt) in enumerate(conditions):
        for rep in range(cfg.n_replications):
            rows.append(
                {
                    "run_id": run_id,
                    "condition_id": cond_id,
                    "replication": rep,
                    "p_detection": p,
                    "lambda_attack_rate": lam,
                    "alpha_decay_rate": alpha,
                    "n_packets_per_attack": n_pkt,
                    "sim_duration_min": cfg.simulation_duration_min,
                    "random_seed": cfg.base_seed + run_id,
                }
            )
            run_id += 1
    plan = pd.DataFrame(rows)
    plan.to_csv(cfg.experiment_plan_csv, index=False)
    print(f"Conditions: {len(conditions)}")
    print(f"Replications: {cfg.n_replications}")
    print(f"Total runs: {len(plan)}")
    print(f"Plan saved: {cfg.experiment_plan_csv}")
    return plan


class ExperimentRunner:
    def __init__(self, plan_csv: str, output_dir: str):
        self.plan = pd.read_csv(plan_csv)
        self.results_file = os.path.join(output_dir, "all_results.csv")
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(self.results_file):
            existing = pd.read_csv(self.results_file)
            self.completed = set(existing["run_id"].tolist())
        else:
            self.completed = set()

    def run_single(self, row: pd.Series):
        cfg = {
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
        sim = NetworkAttackSimulation(cfg)
        sim.run(duration_minutes=float(row["sim_duration_min"]), seed=int(row["random_seed"]))
        summary = sim.get_run_summary()
        summary["attack_counts_per_interval"] = json.dumps(summary["attack_counts_per_interval"])
        summary["inter_arrival_times"] = json.dumps(summary["inter_arrival_times"])
        summary["throughput_timeseries"] = json.dumps(summary["throughput_timeseries"])
        return summary

    def run_all(self, save_every: int = 50):
        remain = self.plan[~self.plan["run_id"].isin(self.completed)]
        print(f"Running {len(remain)} experiments ({len(self.completed)} already completed)")
        new_rows = []
        t0 = time.time()
        for i, (_, row) in enumerate(remain.iterrows(), start=1):
            result = self.run_single(row)
            new_rows.append({**row.to_dict(), **result})
            if i % save_every == 0:
                self._save(new_rows)
                new_rows = []
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-9)
                print(f"Run {i}/{len(remain)} | {rate:.2f} runs/sec")
        if new_rows:
            self._save(new_rows)

    def _save(self, rows):
        new_df = pd.DataFrame(rows)
        if os.path.exists(self.results_file):
            old = pd.read_csv(self.results_file)
            out = pd.concat([old, new_df], ignore_index=True)
        else:
            out = new_df
        out.to_csv(self.results_file, index=False)


def _json_list(cell):
    try:
        return json.loads(cell) if isinstance(cell, str) else []
    except Exception:
        return []


def analyze_h1(df, out_tables, out_figs, show):
    rows = []
    for p in sorted(df["p_detection"].unique()):
        for n in sorted(df["n_packets_per_attack"].unique()):
            sub = df[(df["p_detection"] == p) & (df["n_packets_per_attack"] == n)]
            if len(sub) == 0:
                continue
            theory = 1 - (1 - p) ** n
            obs = sub["observed_detection_rate"].values
            t_stat, p_val = stats.ttest_1samp(obs, theory)
            rows.append({"p": p, "n": n, "theoretical": theory, "observed_mean": float(np.mean(obs)), "p_value": float(p_val)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_tables, "h1_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    n_range = np.arange(1, int(df["n_packets_per_attack"].max()) + 5)
    for p in sorted(df["p_detection"].unique()):
        theory_curve = 1 - (1 - p) ** n_range
        ax.plot(n_range, theory_curve, linewidth=2, label=f"Theory p={p:.2f}")
        for n in sorted(df["n_packets_per_attack"].unique()):
            sub = df[(df["p_detection"] == p) & (df["n_packets_per_attack"] == n)]
            m = sub["observed_detection_rate"].mean()
            se = sub["observed_detection_rate"].std(ddof=1) / np.sqrt(max(len(sub), 1))
            ax.errorbar(n, m, yerr=1.96 * se, fmt="o", color=ax.lines[-1].get_color(), capsize=3)
    ax.set_title("Hypothesis 1: Theoretical vs. Simulated Detection Probability", fontweight="bold")
    ax.set_xlabel("Packets per Attack (n)")
    ax.set_ylabel("P(Detect at Least 1 Packet)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_figs, "h1_detection_overlay.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def analyze_h2(df, out_tables, out_figs, show):
    rows = []
    lambdas = sorted(df["lambda_attack_rate"].unique())
    fig, axes = plt.subplots(1, len(lambdas), figsize=(5 * len(lambdas), 5), squeeze=False)
    axes = axes[0]
    for ax, lam in zip(axes, lambdas):
        sub = df[df["lambda_attack_rate"] == lam]
        duration = float(sub["sim_duration_min"].iloc[0])
        mu = lam * duration
        counts = sub["total_attacks"].values
        max_k = int(max(np.max(counts), mu + 4 * np.sqrt(mu)))
        bins = np.arange(-0.5, max_k + 1.5)
        ax.hist(counts, bins=bins, density=True, alpha=0.7, edgecolor="black", label="Simulated")
        k = np.arange(0, max_k + 1)
        pmf = stats.poisson.pmf(k, mu)
        ax.plot(k, pmf, "ro-", linewidth=2, markersize=5, label=f"Poisson({mu:.0f})")
        ax.set_title(f"lambda = {lam}", fontweight="bold")
        ax.set_xlabel("Number of attacks")
        ax.set_ylabel("Probability")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        obs_freq, _ = np.histogram(counts, bins=np.arange(0, max_k + 2))
        exp_freq = np.array([stats.poisson.pmf(i, mu) * len(counts) for i in range(len(obs_freq))])
        if exp_freq.sum() > 0:
            exp_freq = exp_freq * (obs_freq.sum() / exp_freq.sum())
        try:
            chi2, p_val = stats.chisquare(obs_freq, exp_freq)
        except Exception:
            chi2, p_val = np.nan, np.nan
        rows.append({"lambda": lam, "expected_mean": mu, "observed_mean": float(np.mean(counts)), "chi2": chi2, "p_value": p_val})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_tables, "h2_results.csv"), index=False)
    plt.suptitle("Hypothesis 2: Observed vs. Poisson Attack Counts", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_figs, "h2_poisson_overlay.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def analyze_h3(df, out_tables, out_figs, show):
    rows = []
    lambdas = sorted(df["lambda_attack_rate"].unique())
    fig, axes = plt.subplots(2, len(lambdas), figsize=(5 * len(lambdas), 9), squeeze=False)
    for col, lam in enumerate(lambdas):
        sub = df[df["lambda_attack_rate"] == lam]
        all_ia = []
        for v in sub["inter_arrival_times"].tolist():
            all_ia.extend(_json_list(v))
        ia = np.array(all_ia, dtype=float)
        if len(ia) < 10:
            continue
        ks_stat, p_val = stats.kstest(ia, "expon", args=(0, 1.0 / lam))
        rows.append(
            {
                "lambda": lam,
                "theoretical_mean": 1.0 / lam,
                "observed_mean": float(np.mean(ia)),
                "ks_statistic": float(ks_stat),
                "p_value": float(p_val),
                "n_samples": int(len(ia)),
            }
        )
        ax_h = axes[0, col]
        ax_h.hist(ia, bins=40, density=True, alpha=0.7, edgecolor="black", color="steelblue")
        t_range = np.linspace(0, np.percentile(ia, 99), 200)
        ax_h.plot(t_range, lam * np.exp(-lam * t_range), "r-", linewidth=2.2, label=f"Exp(lambda={lam})")
        ax_h.set_title(f"lambda = {lam}", fontweight="bold")
        ax_h.set_xlabel("Inter-arrival time (min)")
        ax_h.set_ylabel("Density")
        ax_h.legend(fontsize=9)
        ax_h.grid(True, alpha=0.3)

        ax_q = axes[1, col]
        probs = np.linspace(0.01, 0.99, min(len(ia), 200))
        theo_q = stats.expon.ppf(probs, scale=1.0 / lam)
        samp_q = np.quantile(ia, probs)
        ax_q.scatter(theo_q, samp_q, alpha=0.5, s=10)
        m = max(theo_q.max(), samp_q.max())
        ax_q.plot([0, m], [0, m], "r--", linewidth=2, label="Perfect fit")
        ax_q.set_title(f"Q-Q Plot: lambda={lam}", fontweight="bold")
        ax_q.set_xlabel("Theoretical quantiles")
        ax_q.set_ylabel("Sample quantiles")
        ax_q.legend(fontsize=9)
        ax_q.grid(True, alpha=0.3)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_tables, "h3_results.csv"), index=False)
    plt.suptitle("Hypothesis 3: Exponential Inter-Arrival Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_figs, "h3_exponential_analysis.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _decay(t, t0, alpha):
    return t0 * np.exp(-alpha * t)


def analyze_h4(df, out_tables, out_figs, show):
    rows = []
    alphas = sorted(df["alpha_decay_rate"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes
    colors = ["#377eb8", "#ff7f00", "#e41a1c"]
    for i, alpha in enumerate(alphas):
        sub = df[df["alpha_decay_rate"] == alpha]
        fitted = []
        r2s = []
        for _, row in sub.iterrows():
            ts = _json_list(row["throughput_timeseries"])
            if len(ts) < 5:
                continue
            t = np.array([p[0] for p in ts], dtype=float)
            y = np.array([p[1] for p in ts], dtype=float)
            try:
                popt, _ = curve_fit(_decay, t, y, p0=[100.0, alpha], bounds=([0, 0], [200, 10]), maxfev=4000)
                yhat = _decay(t, *popt)
                ss_res = np.sum((y - yhat) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                fitted.append(float(popt[1]))
                r2s.append(float(r2))
            except Exception:
                continue
        if len(fitted) == 0:
            continue
        mf = float(np.mean(fitted))
        sf = float(np.std(fitted, ddof=1)) if len(fitted) > 1 else 0.0
        rows.append({"alpha_theoretical": alpha, "alpha_fitted_mean": mf, "alpha_fitted_std": sf, "mean_r_squared": float(np.mean(r2s))})

        # Plot a few simulated curves
        for _, row in sub.head(4).iterrows():
            ts = _json_list(row["throughput_timeseries"])
            if len(ts) > 1:
                t = [p[0] for p in ts]
                y = [p[1] for p in ts]
                ax1.plot(t, y, "-", color=colors[i % len(colors)], alpha=0.22, linewidth=0.9)
        t_th = np.linspace(0, 10, 120)
        ax1.plot(t_th, 100.0 * np.exp(-alpha * t_th), "--", color=colors[i % len(colors)], linewidth=2.2, label=f"Theory α={alpha}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_tables, "h4_results.csv"), index=False)
    ax1.set_title("Throughput Degradation Under Attack", fontweight="bold")
    ax1.set_xlabel("Time (minutes)")
    ax1.set_ylabel("Throughput (Mbps)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    if len(out) > 0:
        ax2.errorbar(out["alpha_theoretical"], out["alpha_fitted_mean"], yerr=out["alpha_fitted_std"], fmt="o", color="steelblue", capsize=5)
        m = max(float(out["alpha_theoretical"].max()), float(out["alpha_fitted_mean"].max())) * 1.1
        ax2.plot([0, m], [0, m], "r--", linewidth=2, label="Perfect agreement")
        ax2.legend(fontsize=9)
    ax2.set_title("Fitted vs. Theoretical Decay Rate", fontweight="bold")
    ax2.set_xlabel("Theoretical α")
    ax2.set_ylabel("Fitted α")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_figs, "h4_degradation_analysis.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def analyze_h5(df, out_tables, out_figs, show):
    d = df.copy()
    d["theoretical_detection"] = 1 - (1 - d["p_detection"]) ** d["n_packets_per_attack"]
    d["detection_error"] = d["observed_detection_rate"] - d["theoretical_detection"]
    interaction = d.groupby(["p_detection", "lambda_attack_rate"])["detection_error"].agg(["mean", "std"]).round(4)
    interaction.to_csv(os.path.join(out_tables, "h5_interaction_table.csv"))
    print("\nMean detection error by attack rate and IDS quality:")
    print(interaction.to_string())

    pivot = d.groupby(["p_detection", "lambda_attack_rate"])["detection_error"].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-0.12, vmax=0.12)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.1f}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{p:.2f}" for p in pivot.index])
    ax.set_xlabel("Attack Rate λ (attacks/min)")
    ax.set_ylabel("Detection Probability p")
    ax.set_title("Where Models Break Down\nPrediction Error (Observed - Theoretical)", fontweight="bold")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = float(pivot.values[i, j])
            txt = "white" if abs(val) > 0.07 else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", color=txt, fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Prediction Error", shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_figs, "h5_error_heatmap.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    n_focus = 10
    if n_focus not in set(d["n_packets_per_attack"].unique().tolist()):
        n_focus = int(sorted(d["n_packets_per_attack"].unique())[0])
    p_d = d[d["n_packets_per_attack"] == n_focus]
    colors = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]
    for i, p in enumerate(sorted(p_d["p_detection"].unique())):
        sub = p_d[p_d["p_detection"] == p]
        means = sub.groupby("lambda_attack_rate")["observed_detection_rate"].mean()
        stds = sub.groupby("lambda_attack_rate")["observed_detection_rate"].std()
        ncount = sub.groupby("lambda_attack_rate")["observed_detection_rate"].count()
        yerr = 1.96 * stds.values / np.sqrt(np.maximum(ncount.values, 1))
        ax.errorbar(means.index, means.values, yerr=yerr, fmt="o-", color=colors[i % len(colors)], linewidth=2, markersize=7, capsize=4, label=f"p = {p:.2f}")
    ax.set_xlabel("Attack Rate λ (attacks/min)")
    ax.set_ylabel(f"Mean Detection Rate (n={n_focus})")
    ax.set_title("Interaction: IDS Quality × Attack Rate", fontweight="bold")
    ax.set_ylim(0.55, 1.05)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.legend(title="IDS Quality (p)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_figs, "h5_interaction_plot.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def run_full_analysis(cfg: Phase3Config):
    results_file = os.path.join(cfg.results_dir, "all_results.csv")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"{results_file} not found")
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} rows for analysis")
    out_tables = os.path.join(cfg.analysis_dir, "tables")
    out_figs = os.path.join(cfg.analysis_dir, "figures")
    os.makedirs(out_tables, exist_ok=True)
    os.makedirs(out_figs, exist_ok=True)
    analyze_h1(df, out_tables, out_figs, cfg.show_figures)
    analyze_h2(df, out_tables, out_figs, cfg.show_figures)
    analyze_h3(df, out_tables, out_figs, cfg.show_figures)
    analyze_h4(df, out_tables, out_figs, cfg.show_figures)
    analyze_h5(df, out_tables, out_figs, cfg.show_figures)
    print("Analysis complete.")


def main():
    cfg = Phase3Config()
    ensure_dirs(cfg)
    if cfg.run_synthetic:
        generate_synthetic_data(cfg, os.path.join(cfg.results_dir, "all_results.csv"))
    build_experiment_plan(cfg)
    if cfg.run_experiments:
        runner = ExperimentRunner(cfg.experiment_plan_csv, cfg.results_dir)
        runner.run_all()
    if cfg.run_analysis:
        run_full_analysis(cfg)


if __name__ == "__main__":
    main()
