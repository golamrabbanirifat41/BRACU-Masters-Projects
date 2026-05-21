# BRACU Masters Projects

This repository collects two master-level research projects on XAUUSD (Gold vs. US Dollar) direction forecasting using deep learning and federated learning.

## Projects

### 1. Daily XAUUSD Price Direction Forecasting using LSTM & GRU with SHAP Explainability

Location: `Daily-XAUUSD-price-direction-Forecasting-using-LSTM-&-GRU-with-SHAP-explainability`

This project trains recurrent neural network classifiers (LSTM and GRU) to predict daily XAUUSD direction. It includes feature engineering, sequence modeling, validation/test performance metrics, and SHAP explainability for model interpretation.

Key files:
- `train_models.py` — train LSTM and GRU models on engineered features
- `shap_analysis.py` — generate SHAP explainability plots and feature importance
- `xauusd_features.csv` — input feature dataset
- `requirements.txt` — required Python packages
- `results/` — saved model artifacts and metrics
- `figures/cse710/` — evaluation and SHAP visualization figures

### 2. Federated Learning for XAUUSD Direction Forecasting

Location: `Federated-Learning-for-XAUUSD-Direction-Forecasting`

This project implements a federated learning experiment using Flower and PyTorch for XAUUSD direction modeling. It partitions the dataset into simulated clients, trains a global model with FedAvg, and evaluates performance on a held-out test split.

Key files:
- `federated_xauusd.py` — main federated training and evaluation script
- `xauusd_features.csv` — input dataset
- `figures/` — generated experiment plots
- `results/` — saved results and metrics
- `run_output.txt` — sample execution output
- `ieee/` — paper and LaTeX resources documenting the experiment

## Usage

Each project has its own working directory. From the repository root, navigate to the desired project folder and follow its README instructions.

Example:

```bash
cd "Daily-XAUUSD-price-direction-Forecasting-using-LSTM-&-GRU-with-SHAP-explainability"
python train_models.py
```

```bash
cd "Federated-Learning-for-XAUUSD-Direction-Forecasting"
python federated_xauusd.py
```

## Notes

- The LSTM/GRU project focuses on sequence modeling and SHAP explainability.
- The federated learning project demonstrates decentralized training using Flower's FedAvg strategy.
- Both experiments use the same XAUUSD feature dataset format and binary direction target.

## Purpose

This workspace is intended for academic research, experimentation, and comparative analysis of centralized deep learning versus federated approaches for financial signal forecasting.
