# Phase 3 Analysis Report

## Dataset and Run Summary
- Input plan file: `experiment_plan.csv` / `experiment_plan_fixed.csv`
- Results file: `results/all_results.csv` / `results_fixed/all_results.csv`
- Total conditions: `240`
- Replications per condition: `30`
- Total runs: `7,200`

## Hypothesis Outcomes
- **H1 (Binomial detection):** Supported (matches theoretical behavior under tested ranges).
- **H2 (Poisson counts):** Supported (attack counts align with Poisson expectation).
- **H3 (Exponential inter-arrivals):** Supported after inter-arrival measurement fix.
- **H4 (Throughput decay):** Supported (fitted decay close to configured alpha; strong fit quality).
- **H5 (Cross-model consistency):** Supported after excluding undefined zero-attack runs in error metric.

## Statistical Methods Used
- H1: One-sample t-test, confidence intervals, effect size summary
- H2: Chi-square goodness-of-fit test
- H3: Kolmogorov-Smirnov test + Q-Q diagnostics
- H4: Nonlinear exponential model fit (`curve_fit`), bias and `R^2`
- H5: Grouped prediction-error interaction analysis + heatmap

## Key Corrections Applied During Phase 3
- Excluded `total_attacks == 0` runs where detection-rate metrics are undefined.
- Corrected inter-arrival validation input to avoid finite-window bias.
- Aligned metric naming with Phase 3 schema for consistent analysis columns.
- Improved plot readability and consistency for flat/near-saturated conditions.

## Files Generated
- Tables: `analysis/tables/*.csv` (or `analysis_fixed/tables/*.csv`)
- Figures: `analysis/figures/*.png` (or `analysis_fixed/figures/*.png`)
- CI artifacts: workflow uploads CSV and PNG outputs.

## Conclusion
The Phase 3 pipeline executes end-to-end, collects the required data dictionary fields, applies appropriate statistical tests per hypothesis, and produces report-ready figures/tables for mentor review.
