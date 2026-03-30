# Phase 3 Design Document

## Research Question
Does the network attack simulation match the theoretical Binomial, Poisson, Exponential, and throughput degradation models under controlled parameter variations?

## Hypotheses
- **H1 (Binomial detection)**  
  H0: Observed attack-level detection rate equals `1 - (1 - p)^n`.  
  H1: Observed attack-level detection rate differs from `1 - (1 - p)^n`.
- **H2 (Poisson attack counts)**  
  H0: Attack counts follow `Poisson(lambda * t)`.  
  H1: Attack counts do not follow `Poisson(lambda * t)`.
- **H3 (Exponential inter-arrivals)**  
  H0: Inter-arrival times follow `Exp(lambda)`.  
  H1: Inter-arrival times do not follow `Exp(lambda)`.
- **H4 (Throughput degradation)**  
  H0: Throughput follows `T(t) = T0 * exp(-alpha * t)`.  
  H1: Throughput deviates from the exponential decay model.
- **H5 (Cross-model consistency)**  
  H0: Combined prediction error shows no meaningful interaction across conditions.  
  H1: Combined prediction error shows interaction effects.

## Parameter Levels
- `p_detection`: **`[0.30, 0.50, 0.70, 0.85, 0.95]`** (H1 binomial factorial; `GUIDE_FACTORIAL_DETECTION_PROBS`) — × 4 λ × 3 α × 4 n = **240** conditions × 30 = **7,200** runs
- `lambda_attack_rate`: `[0.2, 0.5, 1.0, 2.0]`
- `alpha_decay_rate`: `[0.1, 0.3, 0.5]`
- `n_packets_per_attack`: `[10, 25, 50, 100]`
- `simulation_duration_min`: `10`
- `n_replications`: `30`

## Experiment Size
- Conditions: `5 * 4 * 3 * 4 = 240`
- Replications per condition: `30`
- Total runs: `7,200`

## Response Variables
`run_id`, `condition_id`, `replication`, `p_detection`, `lambda_attack_rate`, `alpha_decay_rate`, `n_packets_per_attack`, `sim_duration_min`, `random_seed`, `observed_detection_rate`, `observed_packet_detection_rate`, `total_attacks`, `total_detected`, `mean_inter_arrival_time`, `attack_counts_per_interval`, `inter_arrival_times`, `final_throughput`, `throughput_timeseries`, `mean_throughput`, `initial_throughput`.

## Analysis Plan
- H1: One-sample t-test + confidence intervals
- H2: Chi-square goodness-of-fit
- H3: Kolmogorov-Smirnov test + Q-Q plot
- H4: Nonlinear exponential curve fit (`curve_fit`) + fit quality metrics
- H5: Interaction/prediction-error summary + heatmap

## Output Locations
- Experiment plan: `experiment_plan.csv` (or `experiment_plan_fixed.csv`)
- Full results dataset: `results/all_results.csv` (or `results_fixed/all_results.csv`)
- Run log: `results/experiment_log.txt` (or `results_fixed/experiment_log.txt`)
- Analysis tables: `analysis/tables/` (or `analysis_fixed/tables/`)
- Analysis figures: `analysis/figures/` (or `analysis_fixed/figures/`)
