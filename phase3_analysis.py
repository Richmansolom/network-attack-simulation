"""
Phase 3 Analysis Script -- Network Attack Simulation Research
=============================================================

Base analysis pipeline with poster-style figures integrated.
Parameters below are the active project parameters.
"""

# ============================================================================
# CONFIGURATION -- MODIFY THIS BLOCK
# ============================================================================

# File paths
RESULTS_FILE = "results/all_results.csv"
OUTPUT_DIR = "analysis"
FIGURES_DIR = f"{OUTPUT_DIR}/figures"
TABLES_DIR = f"{OUTPUT_DIR}/tables"

# Experiment parameters (keep as provided for this project)
DETECTION_PROBS = [0.30, 0.50, 0.70, 0.85, 0.95]
ATTACK_RATES = [0.2, 0.5, 1.0, 2.0, 5.0]
DECAY_RATES = [0.1, 0.3, 0.5]
PACKETS_PER_ATTACK = [1, 2, 3, 5, 10, 25, 50]
N_REPLICATIONS = 30
SIMULATION_DURATION_MIN = 10.0
BASE_SEED = 42

# Initial throughput
T0 = 100.0

# Statistical thresholds
ALPHA = 0.05
N_HYPOTHESES = 5
ALPHA_CORRECTED = ALPHA / N_HYPOTHESES

# Set True only for dry-run pipeline testing
USE_SYNTHETIC_DATA = False


# ============================================================================
# IMPORTS
# ============================================================================

import itertools
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")
np.random.seed(BASE_SEED)

# CLI flags for PowerShell usage:
#   --generate-only       -> generate synthetic data and exit
#   --use-synthetic       -> run full pipeline using synthetic data
#   --results-file <path> -> override input/output CSV path
GENERATE_ONLY = "--generate-only" in sys.argv
if "--use-synthetic" in sys.argv or GENERATE_ONLY:
    USE_SYNTHETIC_DATA = True
if "--results-file" in sys.argv:
    idx = sys.argv.index("--results-file")
    if idx + 1 < len(sys.argv):
        RESULTS_FILE = sys.argv[idx + 1]

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 13
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 13

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

print("=" * 70)
print("PHASE 3 STATISTICAL ANALYSIS")
print("Network Attack Simulation Research")
print("=" * 70)


# ============================================================================
# SECTION 0: SYNTHETIC DATA GENERATOR
# ============================================================================

def generate_synthetic_data(
    detection_probs=DETECTION_PROBS,
    attack_rates=ATTACK_RATES,
    decay_rates=DECAY_RATES,
    packets_list=PACKETS_PER_ATTACK,
    n_reps=N_REPLICATIONS,
    duration=SIMULATION_DURATION_MIN,
    t0_val=T0,
    base_seed=BASE_SEED,
    out_csv=RESULTS_FILE,
):
    rng = np.random.default_rng(base_seed)
    rows = []
    run_id = 0

    conditions = list(itertools.product(detection_probs, attack_rates, decay_rates, packets_list))

    for cond_id, (p, lam, alpha, n_pkt) in enumerate(conditions):
        for rep in range(n_reps):
            seed = base_seed + run_id
            rng_run = np.random.default_rng(seed)

            expected_attacks = max(1, int(rng_run.poisson(lam * duration)))
            detections = 0
            total_packets_detected = 0
            for _ in range(expected_attacks):
                pkts = rng_run.binomial(n_pkt, p)
                total_packets_detected += pkts
                if pkts >= 1:
                    detections += 1

            obs_det_rate = detections / expected_attacks
            obs_pkt_det = total_packets_detected / (expected_attacks * n_pkt)

            counts_per_min = [int(rng_run.poisson(lam)) for _ in range(int(duration))]

            if expected_attacks > 1:
                inter_arrivals = rng_run.exponential(1.0 / lam, size=expected_attacks - 1).tolist()
            else:
                inter_arrivals = [float(rng_run.exponential(1.0 / lam))]

            times = np.linspace(0, duration, 20)
            throughput = t0_val * np.exp(-alpha * times)
            throughput += rng_run.normal(0, 1.5, size=len(times))
            throughput = np.clip(throughput, 0, t0_val)

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
                    "observed_detection_rate": obs_det_rate,
                    "observed_packet_detection_rate": float(np.clip(obs_pkt_det, 0, 1)),
                    "total_attacks": expected_attacks,
                    "total_detected": detections,
                    "mean_inter_arrival_time": float(np.mean(inter_arrivals)),
                    "attack_counts_per_interval": json.dumps(counts_per_min),
                    "inter_arrival_times": json.dumps([round(x, 4) for x in inter_arrivals]),
                    "final_throughput": float(throughput[-1]),
                    "throughput_timeseries": json.dumps(
                        [[round(t, 2), round(v, 2)] for t, v in zip(times.tolist(), throughput.tolist())]
                    ),
                    "mean_throughput": float(np.mean(throughput)),
                    "initial_throughput": t0_val,
                }
            )
            run_id += 1

    df_out = pd.DataFrame(rows)
    out_dir = os.path.dirname(out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"Generated {len(df_out)} synthetic runs across {len(conditions)} conditions")
    print(f"Synthetic file written: {out_csv}")
    return df_out


# ============================================================================
# SECTION 1: LOAD AND VALIDATE DATA
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 1: DATA LOADING AND VALIDATION")
print("=" * 70)

if USE_SYNTHETIC_DATA:
    print("Using SYNTHETIC data (pipeline test)")
    df = generate_synthetic_data(out_csv=RESULTS_FILE)
else:
    print(f"Loading: {RESULTS_FILE}")
    df = pd.read_csv(RESULTS_FILE)

if GENERATE_ONLY:
    print("\nGeneration-only mode complete. Exiting before analysis sections.")
    raise SystemExit(0)

print(f"Total rows: {len(df)}")
print(f"Unique conditions: {df['condition_id'].nunique()}")
reps_per = df.groupby("condition_id").size()
print(f"Replications per condition: min={reps_per.min()}, max={reps_per.max()}, median={reps_per.median():.0f}")

missing = df.isnull().sum()
if missing.sum() > 0:
    print("\nWARNING: Missing values detected:")
    print(missing[missing > 0])
else:
    print("No missing values detected.")

print("\nSanity checks:")
print(f"  Detection rate range: [{df['observed_detection_rate'].min():.3f}, {df['observed_detection_rate'].max():.3f}]")
print(f"  Attack count range: [{df['total_attacks'].min()}, {df['total_attacks'].max()}]")
print(f"  Final throughput range: [{df['final_throughput'].min():.1f}, {df['final_throughput'].max():.1f}]")


# ============================================================================
# SECTION 3: HYPOTHESIS 1 -- BINOMIAL DETECTION
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 3: HYPOTHESIS 1 -- BINOMIAL DETECTION MODEL")
print("H0: Simulated detection rate = 1 - (1-p)^n")
print("=" * 70)

results_h1 = []
for p in sorted(df["p_detection"].unique()):
    for n in sorted(df["n_packets_per_attack"].unique()):
        subset = df[(df["p_detection"] == p) & (df["n_packets_per_attack"] == n)]
        if len(subset) == 0:
            continue

        p_theory = 1 - (1 - p) ** n
        observed = subset["observed_detection_rate"].values
        obs_mean = np.mean(observed)
        obs_std = np.std(observed, ddof=1)

        t_stat, p_val = stats.ttest_1samp(observed, p_theory)
        se = obs_std / np.sqrt(len(observed))
        ci_lower = obs_mean - 1.96 * se
        ci_upper = obs_mean + 1.96 * se
        theory_in_ci = ci_lower <= p_theory <= ci_upper
        d = (obs_mean - p_theory) / obs_std if obs_std > 0 else 0

        results_h1.append(
            {
                "p": p,
                "n": n,
                "theoretical": round(p_theory, 4),
                "observed_mean": round(obs_mean, 4),
                "observed_std": round(obs_std, 4),
                "difference": round(obs_mean - p_theory, 4),
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
                "theory_in_ci": theory_in_ci,
                "t_statistic": round(t_stat, 4),
                "p_value": round(p_val, 4),
                "cohens_d": round(d, 3),
                "significant": p_val < ALPHA_CORRECTED,
            }
        )

        status = "MATCH" if p_val > ALPHA_CORRECTED else "DIFFERS*"
        print(f"  p={p:.2f}, n={n:3d}: theory={p_theory:.4f}, obs={obs_mean:.4f}, p-val={p_val:.4f}, d={d:+.3f} [{status}]")

h1_table = pd.DataFrame(results_h1)
h1_table.to_csv(f"{TABLES_DIR}/h1_results.csv", index=False)

# Poster-style Figure 1
fig, ax = plt.subplots(figsize=(10, 7))
n_range = np.arange(1, max(PACKETS_PER_ATTACK) + 5)
colors = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]

for i, p in enumerate(sorted(df["p_detection"].unique())):
    theory_curve = 1 - (1 - p) ** n_range
    ax.plot(n_range, theory_curve, "-", color=colors[i % len(colors)], linewidth=2.5, label=f"Theory p={p:.2f}", zorder=2)
    for n in sorted(df["n_packets_per_attack"].unique()):
        sub = df[(df["p_detection"] == p) & (df["n_packets_per_attack"] == n)]
        if len(sub) == 0:
            continue
        m = sub["observed_detection_rate"].mean()
        se = sub["observed_detection_rate"].std() / np.sqrt(len(sub))
        ax.errorbar(
            n,
            m,
            yerr=1.96 * se,
            fmt="o",
            color=colors[i % len(colors)],
            markersize=7,
            capsize=3,
            capthick=1.5,
            markeredgecolor="black",
            markeredgewidth=0.5,
            zorder=3,
        )

ax.set_xlabel("Packets per Attack (n)", fontsize=14)
ax.set_ylabel("P(Detect >= 1 Packet)", fontsize=14)
ax.set_title("Theoretical vs. Simulated Detection Probability", fontsize=16, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/h1_detection_overlay.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\nFigure saved: {FIGURES_DIR}/h1_detection_overlay.png")


# ============================================================================
# SECTION 4: HYPOTHESIS 2 -- POISSON ATTACK COUNTS
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 4: HYPOTHESIS 2 -- POISSON ATTACK COUNTS")
print("H0: Attack counts ~ Poisson(lambda * t)")
print("=" * 70)

results_h2 = []
n_lambdas = len(df["lambda_attack_rate"].unique())
fig, axes = plt.subplots(1, n_lambdas, figsize=(4.5 * n_lambdas, 5), squeeze=False)
axes = axes[0]

for ax, lam in zip(axes, sorted(df["lambda_attack_rate"].unique())):
    subset = df[df["lambda_attack_rate"] == lam]
    duration = subset["sim_duration_min"].iloc[0]
    expected_mean = lam * duration
    counts = subset["total_attacks"].values

    max_k = int(max(counts.max(), expected_mean + 4 * np.sqrt(expected_mean)))
    bins = np.arange(-0.5, max_k + 1.5)
    ax.hist(counts, bins=bins, density=True, alpha=0.7, edgecolor="black", color="steelblue", label="Simulated")
    k_vals = np.arange(0, max_k + 1)
    ax.plot(k_vals, stats.poisson.pmf(k_vals, expected_mean), "ro-", markersize=5, linewidth=2, label=f"Poisson({expected_mean:.0f})")
    ax.set_xlabel("Number of Attacks")
    ax.set_ylabel("Probability")
    ax.set_title(f"lambda = {lam}", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    obs_freq, _ = np.histogram(counts, bins=np.arange(0, max_k + 2))
    exp_freq = np.array([stats.poisson.pmf(k, expected_mean) * len(counts) for k in range(len(obs_freq))])
    if exp_freq.sum() > 0:
        exp_freq = exp_freq * obs_freq.sum() / exp_freq.sum()

    obs_c, exp_c, oa, ea = [], [], 0, 0
    for o, e in zip(obs_freq, exp_freq):
        oa += o
        ea += e
        if ea >= 5:
            obs_c.append(oa)
            exp_c.append(ea)
            oa = 0
            ea = 0
    if oa > 0 or ea > 0:
        if obs_c:
            obs_c[-1] += oa
            exp_c[-1] += ea
        else:
            obs_c.append(oa)
            exp_c.append(ea)

    if len(obs_c) >= 2:
        chi2, p_val = stats.chisquare(obs_c, exp_c)
    else:
        chi2, p_val = 0, 1.0

    results_h2.append(
        {
            "lambda": lam,
            "expected_mean": expected_mean,
            "observed_mean": round(np.mean(counts), 2),
            "observed_std": round(np.std(counts, ddof=1), 2),
            "chi2": round(chi2, 3),
            "p_value": round(p_val, 4),
            "significant": p_val < ALPHA_CORRECTED,
        }
    )

    status = "MATCHES POISSON" if p_val > ALPHA_CORRECTED else "DEVIATES*"
    print(f"  lambda={lam:.1f}: E[N]={expected_mean:.1f}, obs_mean={np.mean(counts):.2f}, chi2={chi2:.2f}, p={p_val:.4f} [{status}]")

h2_table = pd.DataFrame(results_h2)
h2_table.to_csv(f"{TABLES_DIR}/h2_results.csv", index=False)

plt.suptitle("H2: Observed vs. Poisson Attack Counts", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/h2_poisson_overlay.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\nFigure saved: {FIGURES_DIR}/h2_poisson_overlay.png")


# ============================================================================
# SECTION 5: HYPOTHESIS 3 -- EXPONENTIAL INTER-ARRIVALS
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 5: HYPOTHESIS 3 -- EXPONENTIAL INTER-ARRIVAL TIMES")
print("H0: Inter-arrival times ~ Exponential(lambda)")
print("=" * 70)

results_h3 = []
n_lam = len(df["lambda_attack_rate"].unique())
fig, axes = plt.subplots(2, n_lam, figsize=(4.5 * n_lam, 9), squeeze=False)

for col, lam in enumerate(sorted(df["lambda_attack_rate"].unique())):
    subset = df[df["lambda_attack_rate"] == lam]
    all_ia = []
    for _, row in subset.iterrows():
        try:
            all_ia.extend(json.loads(row["inter_arrival_times"]))
        except Exception:
            continue
    ia = np.array(all_ia)
    if len(ia) < 10:
        print(f"  lambda={lam}: Too few inter-arrival times ({len(ia)})")
        continue

    ks_stat, p_val = stats.kstest(ia, "expon", args=(0, 1.0 / lam))
    results_h3.append(
        {
            "lambda": lam,
            "theoretical_mean": round(1.0 / lam, 3),
            "observed_mean": round(np.mean(ia), 3),
            "theoretical_median": round(np.log(2) / lam, 3),
            "observed_median": round(np.median(ia), 3),
            "ks_statistic": round(ks_stat, 4),
            "p_value": round(p_val, 4),
            "n_samples": len(ia),
            "significant": p_val < ALPHA_CORRECTED,
        }
    )
    status = "MATCHES EXP" if p_val > ALPHA_CORRECTED else "DEVIATES*"
    print(f"  lambda={lam:.1f}: E[T]={1.0/lam:.2f}, obs_mean={np.mean(ia):.3f}, KS={ks_stat:.4f}, p={p_val:.4f} [{status}]")

    ax_h = axes[0, col]
    ax_h.hist(ia, bins=50, density=True, alpha=0.7, edgecolor="black", color="steelblue", label="Simulated")
    t_range = np.linspace(0, np.percentile(ia, 99), 200)
    ax_h.plot(t_range, lam * np.exp(-lam * t_range), "r-", linewidth=2.5, label=f"Exp(lambda={lam})")
    ax_h.set_xlabel("Inter-Arrival Time (min)")
    ax_h.set_ylabel("Density")
    ax_h.set_title(f"lambda = {lam}", fontweight="bold")
    ax_h.legend(fontsize=8)
    ax_h.grid(True, alpha=0.3)

    ax_q = axes[1, col]
    n_pts = min(len(ia), 200)
    probs = np.linspace(0.01, 0.99, n_pts)
    theo_q = stats.expon.ppf(probs, scale=1.0 / lam)
    samp_q = np.quantile(ia, probs)
    ax_q.scatter(theo_q, samp_q, alpha=0.5, s=10)
    max_v = max(theo_q.max(), samp_q.max())
    ax_q.plot([0, max_v], [0, max_v], "r--", linewidth=2, label="Perfect fit")
    ax_q.set_xlabel("Theoretical Quantiles")
    ax_q.set_ylabel("Sample Quantiles")
    ax_q.set_title(f"Q-Q: lambda = {lam}", fontweight="bold")
    ax_q.legend(fontsize=8)
    ax_q.grid(True, alpha=0.3)

h3_table = pd.DataFrame(results_h3)
h3_table.to_csv(f"{TABLES_DIR}/h3_results.csv", index=False)

plt.suptitle("H3: Exponential Inter-Arrival Time Analysis", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/h3_exponential_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\nFigure saved: {FIGURES_DIR}/h3_exponential_analysis.png")


# ============================================================================
# SECTION 6: HYPOTHESIS 4 -- THROUGHPUT DEGRADATION
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 6: HYPOTHESIS 4 -- THROUGHPUT DEGRADATION")
print("H0: T(t) = T0 * exp(-alpha * t)")
print("=" * 70)

def decay_model(t, t0_fit, alpha_fit):
    return t0_fit * np.exp(-alpha_fit * t)


results_h4 = []
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax1 = axes[0]
acolors = ["#377eb8", "#ff7f00", "#e41a1c"]

for k, alpha_theory in enumerate(sorted(df["alpha_decay_rate"].unique())):
    subset = df[df["alpha_decay_rate"] == alpha_theory]
    fitted_alphas = []
    r2_values = []

    t = np.linspace(0, SIMULATION_DURATION_MIN, 100)
    ax1.plot(t, T0 * np.exp(-alpha_theory * t), "--", color=acolors[k % len(acolors)], linewidth=2.5, label=f"Theory alpha={alpha_theory}")

    for _, row in subset.iterrows():
        try:
            ts = json.loads(row["throughput_timeseries"])
            times = np.array([pt[0] for pt in ts])
            values = np.array([pt[1] for pt in ts])
        except Exception:
            continue
        if len(times) < 5:
            continue
        try:
            popt, _ = curve_fit(
                decay_model,
                times,
                values,
                p0=[T0, alpha_theory],
                bounds=([0, 0], [200, 10]),
                maxfev=5000,
            )
            pred = decay_model(times, *popt)
            ss_res = np.sum((values - pred) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            fitted_alphas.append(popt[1])
            r2_values.append(r2)
        except RuntimeError:
            continue

    for _, row in subset.head(5).iterrows():
        try:
            ts = json.loads(row["throughput_timeseries"])
            ax1.plot([pt[0] for pt in ts], [pt[1] for pt in ts], "-", color=acolors[k % len(acolors)], alpha=0.25, linewidth=0.9)
        except Exception:
            continue

    if len(fitted_alphas) == 0:
        print(f"  alpha={alpha_theory:.1f}: No successful fits")
        continue

    fa = np.array(fitted_alphas)
    mean_fa = np.mean(fa)
    std_fa = np.std(fa, ddof=1)
    bias = mean_fa - alpha_theory
    mean_r2 = np.mean(r2_values)
    results_h4.append(
        {
            "alpha_theoretical": alpha_theory,
            "alpha_fitted_mean": round(mean_fa, 4),
            "alpha_fitted_std": round(std_fa, 4),
            "bias": round(bias, 4),
            "bias_pct": round(100 * bias / alpha_theory, 1),
            "mean_r_squared": round(mean_r2, 4),
            "n_fits": len(fitted_alphas),
            "half_life_theory": round(np.log(2) / alpha_theory, 2),
            "half_life_fitted": round(np.log(2) / mean_fa, 2),
        }
    )
    print(
        f"  alpha={alpha_theory:.1f}: fitted={mean_fa:.4f} +/- {std_fa:.4f}, "
        f"bias={bias:+.4f} ({100*bias/alpha_theory:+.1f}%), R2={mean_r2:.4f}"
    )

h4_table = pd.DataFrame(results_h4)
h4_table.to_csv(f"{TABLES_DIR}/h4_results.csv", index=False)

ax1.set_xlabel("Time (minutes)", fontsize=13)
ax1.set_ylabel("Throughput (Mbps)", fontsize=13)
ax1.set_title("Throughput Degradation Under Attack", fontsize=14, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-5, 110)

ax2 = axes[1]
if len(h4_table) > 0:
    ax2.errorbar(
        h4_table["alpha_theoretical"],
        h4_table["alpha_fitted_mean"],
        yerr=1.96 * h4_table["alpha_fitted_std"],
        fmt="o",
        markersize=12,
        capsize=6,
        capthick=2,
        color="steelblue",
        markeredgecolor="black",
        linewidth=2,
    )
    mx = max(h4_table["alpha_theoretical"].max(), h4_table["alpha_fitted_mean"].max()) * 1.15
    ax2.plot([0, mx], [0, mx], "r--", linewidth=2, label="Perfect agreement")
ax2.set_xlabel("Theoretical alpha", fontsize=13)
ax2.set_ylabel("Fitted alpha", fontsize=13)
ax2.set_title("Fitted vs. Theoretical Decay Rate", fontsize=14, fontweight="bold")
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/h4_degradation_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\nFigure saved: {FIGURES_DIR}/h4_degradation_analysis.png")


# ============================================================================
# SECTION 7: HYPOTHESIS 5 -- CROSS-VALIDATION
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 7: HYPOTHESIS 5 -- MULTI-MODEL CROSS-VALIDATION")
print("H0: Models remain consistent when applied together")
print("=" * 70)

df["theoretical_detection"] = 1 - (1 - df["p_detection"]) ** df["n_packets_per_attack"]
df["detection_error"] = df["observed_detection_rate"] - df["theoretical_detection"]

print("\nMean detection error by (p, lambda):")
interaction = df.groupby(["p_detection", "lambda_attack_rate"])["detection_error"].agg(["mean", "std"]).round(4)
print(interaction.to_string())
interaction.to_csv(f"{TABLES_DIR}/h5_interaction_table.csv")

# Poster-style Figure 2: Error heatmap (small n only)
small_n_df = df[df["n_packets_per_attack"] <= 5]
pivot = small_n_df.groupby(["p_detection", "lambda_attack_rate"])["detection_error"].mean().unstack()
max_err = max(abs(pivot.values.min()), abs(pivot.values.max()))
vmax = max(max_err * 1.1, 0.02)

fig, ax = plt.subplots(figsize=(9, 6.5))
im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"{v}" for v in pivot.columns], fontsize=12)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=12)
ax.set_xlabel("Attack Rate lambda (attacks/min)", fontsize=14)
ax.set_ylabel("Detection Probability p", fontsize=14)
ax.set_title("Where Models Break Down (n <= 5 packets)\nPrediction Error (Observed - Theoretical)", fontsize=16, fontweight="bold")

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        color = "white" if abs(val) > vmax * 0.6 else "black"
        ax.text(j, i, f"{val:+.3f}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

plt.colorbar(im, ax=ax, shrink=0.85, label="Prediction Error")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/h5_error_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\nFigure saved: {FIGURES_DIR}/h5_error_heatmap.png")

# Poster-style Figure 4: Interaction plot (n=1)
int_data = df[df["n_packets_per_attack"] == 1]
fig, ax = plt.subplots(figsize=(9, 6))
icolors = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]

for i, p in enumerate(sorted(int_data["p_detection"].unique())):
    sub = int_data[int_data["p_detection"] == p]
    means = sub.groupby("lambda_attack_rate")["observed_detection_rate"].mean()
    stds = sub.groupby("lambda_attack_rate")["observed_detection_rate"].std()
    ns = sub.groupby("lambda_attack_rate").size()
    ax.errorbar(
        means.index,
        means.values,
        yerr=1.96 * stds.values / np.sqrt(ns.values),
        fmt="o-",
        color=icolors[i % len(icolors)],
        linewidth=2.5,
        markersize=9,
        capsize=4,
        label=f"p = {p:.2f}",
    )
    ax.axhline(y=p, color=icolors[i % len(icolors)], linestyle=":", alpha=0.4, linewidth=1)

ax.set_xlabel("Attack Rate lambda (attacks/min)", fontsize=14)
ax.set_ylabel("Mean Detection Rate (n=1)", fontsize=14)
ax.set_title("Interaction: IDS Quality x Attack Rate", fontsize=16, fontweight="bold")
ax.legend(title="IDS Quality (p)", fontsize=10, title_fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/h5_interaction_plot.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved: {FIGURES_DIR}/h5_interaction_plot.png")


# ============================================================================
# SECTION 8: EFFECT SIZES
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 8: EFFECT SIZES (Cohen's d)")
print("=" * 70)

print("\nH1 Effect Sizes:")
for _, row in h1_table.iterrows():
    label = (
        "negligible"
        if abs(row["cohens_d"]) < 0.2
        else "small"
        if abs(row["cohens_d"]) < 0.5
        else "medium"
        if abs(row["cohens_d"]) < 0.8
        else "LARGE"
    )
    print(f"  p={row['p']:.2f}, n={row['n']:3.0f}: d = {row['cohens_d']:+.3f} ({label})")


# ============================================================================
# SECTION 9: FOREST PLOT / CONFIDENCE INTERVALS
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 9: CONFIDENCE INTERVAL FOREST PLOT")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, max(8, len(h1_table) * 0.4)))
labels = []
pos = 0

for _, row in h1_table.sort_values(["p", "n"], ascending=[False, True]).iterrows():
    ci_w_lo = row["observed_mean"] - row["ci_lower"]
    ci_w_hi = row["ci_upper"] - row["observed_mean"]
    color = "#2ca02c" if row["theory_in_ci"] else "#d62728"

    ax.errorbar(
        row["observed_mean"],
        pos,
        xerr=[[ci_w_lo], [ci_w_hi]],
        fmt="o",
        color=color,
        markersize=6,
        capsize=3,
        capthick=1.5,
        linewidth=1.5,
    )
    ax.plot(row["theoretical"], pos, "|", color="black", markersize=12, markeredgewidth=2)
    labels.append(f"p={row['p']:.2f}, n={int(row['n'])}")
    pos += 1

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Detection Rate")
ax.set_title(
    "H1 Forest Plot: Observed (dots) vs. Theoretical (bars)\nGreen = CI contains theory, Red = CI excludes theory",
    fontweight="bold",
)
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/h1_forest_plot.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved: {FIGURES_DIR}/h1_forest_plot.png")


# ============================================================================
# SECTION 10: SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 10: SUMMARY")
print("=" * 70)

print(f"\nSignificance threshold: alpha = {ALPHA_CORRECTED:.3f} (Bonferroni-corrected for {N_HYPOTHESES} tests)")

print("\nH1 (Binomial Detection):")
n_sig = h1_table["significant"].sum()
print(f"  {n_sig} of {len(h1_table)} conditions show significant deviation")

print("\nH2 (Poisson Counts):")
for _, row in h2_table.iterrows():
    s = "SIGNIFICANT" if row["significant"] else "not significant"
    print(f"  lambda={row['lambda']:.1f}: {s} (p={row['p_value']:.4f})")

print("\nH3 (Exponential Inter-Arrivals):")
for _, row in h3_table.iterrows():
    s = "SIGNIFICANT" if row["significant"] else "not significant"
    print(f"  lambda={row['lambda']:.1f}: {s} (p={row['p_value']:.4f})")

print("\nH4 (Throughput Degradation):")
for _, row in h4_table.iterrows():
    print(f"  alpha={row['alpha_theoretical']:.1f}: bias={row['bias_pct']:+.1f}%, R2={row['mean_r_squared']:.4f}")

print(f"\nFigures saved to: {FIGURES_DIR}/")
print(f"Tables saved to:  {TABLES_DIR}/")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
