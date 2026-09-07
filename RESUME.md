# Resume positioning

**AI Energy Forecaster — Explainable Compute Planning & Model Validation**
Python · XGBoost · scikit-learn · TreeSHAP · split conformal prediction · Streamlit

Suggested bullets:

- Built a pre-run AI training energy planning app with scenario comparisons, energy-budget screening, exact TreeSHAP explanations, and downloadable evidence reports.
- Audited 12,260 source rows, removed runtime and provenance leakage, and implemented source-isolated evaluation with 2,400 synthetic test rows and 235 external labeled rows.
- Added synthetic split-conformal calibration, model/data hashes, saved preprocessing, a legacy-model comparison, and regression tests; exposed external generalization failure rather than presenting blended R² as real-world accuracy.

For a two-bullet resume, combine the first and third and keep the data-audit bullet. Avoid claims such as “99% accurate,” “trained on four real-world datasets,” “reduced energy by 40%,” or “production-grade energy forecasting.” They are not supported by this project's evidence.

## Two-minute walkthrough

1. Open Forecast. Submit a simulation preset and explain that inputs are known before training starts.
2. Show the actual SHAP drivers and rerun scenarios; point out that fewer GPUs need not imply proportionally less total energy.
3. Open the model card. Contrast corrected synthetic R² 0.9940 with external R² −2.3572, both on log-energy.
4. Explain the source audit: most labels are simulated, HF/MLPerf labels are estimated, and one CodeCarbon CPU run is insufficient to validate GPU training.
5. Show the split manifest, model hashes, archived model, and calibration coverage. Explain why 90% nominal synthetic intervals do not transfer to external data.

## Distinction from industrial predictive maintenance

The central problem is **compute planning under uncertain energy estimates**, with an emphasis on auditing the validity of ML claims. Its interaction is workload comparison and budget screening. The technical depth is in leakage control, saved preprocessing, source separation, uncertainty under distribution shift, and exact attribution. It does not duplicate a maintenance agent or an MLOps platform.

## Honest next step

Collect repeated, instrumented training runs with complete pre-run workload and hardware metadata. Split by workload/hardware groups, compare against a physics baseline, and validate on an untouched measurement set before claiming useful real-world forecasting. More synthetic rows or a GenAI explanation layer would not solve the current validation gap.
