# Daily XAUUSD Price Direction Forecasting using LSTM & GRU with SHAP Explainability

This repository implements a daily direction forecasting pipeline for the XAUUSD (Gold vs. US Dollar) market using recurrent neural networks (LSTM and GRU) and SHAP explainability.

## Project Overview

- `train_models.py`: trains LSTM and GRU sequence classifiers on engineered XAUUSD features.
- `shap_analysis.py`: loads the best trained model and produces SHAP explainability plots for model interpretation.
- `xauusd_features.csv`: input feature dataset with daily gold price signals and target direction labels.
- `results/`: stores trained model artifacts, evaluation metrics, and SHAP importance outputs.
- `figures/cse710/`: stores generated model evaluation and SHAP visualization figures.

## Requirements

Install the required Python dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The main dependencies are:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `joblib`
- `tensorflow`
- `shap`

## Usage

### 1. Train models

Run the training script to build LSTM and GRU models on the historical XAUUSD dataset:

```bash
python train_models.py
```

This script will:

- load `xauusd_features.csv`
- scale features with `StandardScaler`
- create rolling lookback sequences
- train LSTM and GRU models with early stopping
- save evaluation metrics to `results/cse710_metrics.csv` and `results/cse710_metrics.json`
- save the best model to `results/cse710_best_model.keras`
- save a bundled model object to `results/cse710_best_model.pkl`
- save training plots and evaluation figures to `figures/cse710/`

### 2. Generate SHAP explainability

After training, run the SHAP analysis script:

```bash
python shap_analysis.py
```

This script will:

- load the best saved model bundle from `results/cse710_best_model.pkl`
- compute SHAP values for the test set
- generate summary and dependence plots under `figures/cse710/`
- export aggregated SHAP importance to `results/cse710_shap_importance.csv`

## Output Files

Important generated files include:

- `results/cse710_best_model.keras` — best Keras model file
- `results/cse710_best_model.pkl` — bundled model, scaler, and test metadata
- `results/cse710_metrics.csv` / `results/cse710_metrics.json` — evaluation metrics for validation and test sets
- `results/cse710_shap_importance.csv` — aggregated SHAP feature importance
- `figures/cse710/` — evaluation and SHAP plots

## Notes

- The model uses a 20-day lookback window by default.
- Training uses a chronological train/validation/test split (70% train, 15% validation, 15% test).
- SHAP explainability is applied to the best recurrent model selected by validation AUC.

## License

This repository is provided for research and academic purposes.
