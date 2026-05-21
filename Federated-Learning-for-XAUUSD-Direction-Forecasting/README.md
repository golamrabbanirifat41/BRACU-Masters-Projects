# Federated Learning for XAUUSD Direction Forecasting

This repository implements a federated learning pipeline for daily XAUUSD (Gold vs. US Dollar) direction forecasting using Flower and PyTorch.

The project reuses the original tabular model and preprocessing pipeline, while migrating the federated orchestration to Flower's FedAvg strategy.

## Project Structure

- `federated_xauusd.py` — main federated training and evaluation script.
- `xauusd_features.csv` — input dataset containing engineered gold price features and direction labels.
- `figures/` — generated plots for training and evaluation.
- `results/` — saved results and metrics from federated training.
- `ieee/` — paper and LaTeX resources for documenting the experiment.
- `run_output.txt` — sample console output from a completed run.

## Dependencies

The script requires the following Python packages:

- `flwr`
- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

If you want to install a virtual environment from scratch, use the Python package manager of your choice:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install flwr torch numpy pandas scikit-learn matplotlib
```

> Note: The exact `torch` package version may depend on your CUDA setup. If you do not use CUDA, install the CPU-only wheel.

## Usage

Run the federated experiment from the project root:

```bash
python federated_xauusd.py
```

This script will:

- load `xauusd_features.csv`
- split the data chronologically into a federated client pool and a global hold-out test set
- partition the pool into 3 simulated clients
- train a PyTorch MLP model using Flower's FedAvg strategy
- evaluate the global model on the held-out test set
- save any generated figures under `figures/`

## Output

The experiment produces:

- evaluation metrics for global and client-local performance
- training and test result plots in `figures/`
- any logged console output in `run_output.txt`

## Notes

- The dataset is split into 85% federated pool and 15% global test set.
- Federated training runs for 30 rounds by default, with 3 local epochs per client.
- The model is a lightweight MLP trained with binary cross-entropy on the XAUUSD direction target.

## Paper and Documentation

The `ieee/` folder contains LaTeX resources and a paper draft documenting the federated learning experiment.

---

This repository is intended for academic research and experimentation with federated learning on financial signal modeling.
