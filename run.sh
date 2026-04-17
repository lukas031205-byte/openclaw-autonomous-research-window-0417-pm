#!/usr/bin/env bash
# Run Experiment D: Cross-Encoder Confound Separation
# Result: pearson_r=-0.8973, spearman_rho=-0.9148, partial_r=-0.8981

python3 experiment_d.py

# To run Experiment E (requires SDXL VAE, ~3GB, CPU slow):
# python3 experiment_e.py
