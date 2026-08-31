#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python evaluate_model.py --n-trials 500 > training_12hr.log 2>&1 &
echo "Started God Mode 12-hour training run in the background."
echo "Logs are being written to training_12hr.log."
echo "To view the dashboard, run: mlflow ui"
