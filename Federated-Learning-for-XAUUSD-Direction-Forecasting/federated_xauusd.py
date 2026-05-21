"""
CSE711: Federated Financial Signal Modeling - XAUUSD direction with Flower.

This script keeps the original PyTorch tabular model and preprocessing, but
migrates the federated orchestration to Flower's FedAvg strategy so the project
uses an FL framework directly.
"""

import json
import os
import random
import threading
import time
import warnings

import flwr as fl
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DATA = "xauusd_features.csv"
FIG = "figures"
RES = "results"
ROUNDS = 30
LOCAL_EPOCHS = 3
LR = 1e-3
BS = 64
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)


def to_float_metrics(metrics):
    return {key: float(value) for key, value in metrics.items()}


# ---------------------------------------------------------------------------
# 1. Load + chronological split: 85% pool for clients, last 15% global test
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
feature_cols = [c for c in df.columns if c not in ("date", "target")]
n = len(df)
split = int(0.85 * n)
df_pool = df.iloc[:split].reset_index(drop=True)
df_test = df.iloc[split:].reset_index(drop=True)
print(
    f"Pool: {len(df_pool)} rows "
    f"({df_pool['date'].iloc[0].date()} -> {df_pool['date'].iloc[-1].date()})"
)
print(
    f"Test: {len(df_test)} rows "
    f"({df_test['date'].iloc[0].date()} -> {df_test['date'].iloc[-1].date()})"
)

# Split pool into 3 equal time-period clients.
idx_splits = np.array_split(np.arange(len(df_pool)), 3)
client_dfs = [df_pool.iloc[ix].reset_index(drop=True) for ix in idx_splits]
for i, cdf in enumerate(client_dfs):
    print(
        f"Client {i}: {len(cdf)} rows  "
        f"{cdf['date'].iloc[0].date()} -> {cdf['date'].iloc[-1].date()}  "
        f"up_rate={cdf['target'].mean():.3f}"
    )

# Shared preprocessing is kept for compatibility with the existing experiment.
scaler = StandardScaler().fit(df_pool[feature_cols].values)


def to_tensors(frame):
    X = scaler.transform(frame[feature_cols].values).astype(np.float32)
    y = frame["target"].values.astype(np.float32)
    return torch.tensor(X), torch.tensor(y)


client_data = [to_tensors(cdf) for cdf in client_dfs]
X_test_t, y_test_t = to_tensors(df_test)
INPUT_DIM = len(feature_cols)
N_CLIENTS = len(client_data)


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def get_params(model):
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def set_params(model, params):
    for p, new in zip(model.parameters(), params):
        p.data = torch.tensor(new, dtype=p.dtype)


def local_train(model, X, y, epochs=3, lr=1e-3, bs=64):
    loader = DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    logits = model(X)
    prob = torch.sigmoid(logits).cpu().numpy()
    pred = (prob >= 0.5).astype(int)
    yt = y.cpu().numpy().astype(int)
    metrics = {
        "accuracy": accuracy_score(yt, pred),
        "precision": precision_score(yt, pred, zero_division=0),
        "recall": recall_score(yt, pred, zero_division=0),
        "f1": f1_score(yt, pred, zero_division=0),
        "roc_auc": roc_auc_score(yt, prob),
        "loss": float(nn.BCEWithLogitsLoss()(logits, y).item()),
    }
    return to_float_metrics(metrics), prob, pred


def weighted_average(metrics):
    if not metrics:
        return {}
    total = sum(num_examples for num_examples, _ in metrics)
    aggregated = {}
    keys = [key for key in metrics[0][1].keys() if key != "cid"]
    for key in keys:
        aggregated[key] = float(
            sum(num_examples * values[key] for num_examples, values in metrics) / total
        )
    return aggregated


history = {"round": [], "global_test": [], "client_local": [[] for _ in range(N_CLIENTS)]}


class XAUUSDClient(fl.client.NumPyClient):
    def __init__(self, cid, X, y):
        self.cid = int(cid)
        self.X = X
        self.y = y
        self.model = MLP(INPUT_DIM)

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        epochs = int(config.get("local_epochs", LOCAL_EPOCHS))
        lr = float(config.get("lr", LR))
        bs = int(config.get("batch_size", BS))
        local_train(self.model, self.X, self.y, epochs=epochs, lr=lr, bs=bs)
        metrics, _, _ = evaluate(self.model, X_test_t, y_test_t)
        metrics["cid"] = self.cid
        return get_params(self.model), len(self.X), metrics

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        metrics, _, _ = evaluate(self.model, self.X, self.y)
        return metrics["loss"], len(self.X), metrics


class TrackingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, tracker, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = tracker
        self.final_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        for _, fit_res in results:
            cid = int(fit_res.metrics["cid"])
            client_metrics = {
                key: float(value)
                for key, value in fit_res.metrics.items()
                if key != "cid"
            }
            self.tracker["client_local"][cid].append(client_metrics)
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics


def get_evaluate_fn(tracker):
    def evaluate_fn(server_round, parameters, config):
        model = MLP(INPUT_DIM)
        set_params(model, parameters)
        metrics, _, _ = evaluate(model, X_test_t, y_test_t)
        tracker["round"].append(server_round)
        tracker["global_test"].append(metrics)
        if server_round == 1 or server_round % 5 == 0 or server_round == ROUNDS:
            print(
                f"Round {server_round:2d} | global test acc={metrics['accuracy']:.4f}  "
                f"AUC={metrics['roc_auc']:.4f}  loss={metrics['loss']:.4f}"
            )
        return metrics["loss"], {
            key: value for key, value in metrics.items() if key != "loss"
        }

    return evaluate_fn


def fit_config(server_round):
    return {
        "local_epochs": LOCAL_EPOCHS,
        "lr": LR,
        "batch_size": BS,
        "round": server_round,
    }


def run_client(server_address, cid, X, y, delay=2.0):
    time.sleep(delay)
    client = XAUUSDClient(cid, X, y).to_client()
    fl.client.start_client(server_address=server_address, client=client)


print("\n--- Federated training with Flower FedAvg (3 clients) ---")
initial_model = MLP(INPUT_DIM)
strategy = TrackingFedAvg(
    tracker=history,
    fraction_fit=1.0,
    min_fit_clients=N_CLIENTS,
    min_available_clients=N_CLIENTS,
    fraction_evaluate=0.0,
    evaluate_fn=get_evaluate_fn(history),
    on_fit_config_fn=fit_config,
    fit_metrics_aggregation_fn=weighted_average,
    initial_parameters=ndarrays_to_parameters(get_params(initial_model)),
)
server_address = "127.0.0.1:8089"
client_threads = []
for cid, (Xc, yc) in enumerate(client_data):
    thread = threading.Thread(
        target=run_client,
        args=(server_address, cid, Xc, yc, 2.0 + 0.2 * cid),
        daemon=True,
    )
    thread.start()
    client_threads.append(thread)

fl.server.start_server(
    server_address=server_address,
    config=fl.server.ServerConfig(num_rounds=ROUNDS),
    strategy=strategy,
)

for thread in client_threads:
    thread.join()

if strategy.final_parameters is None:
    raise RuntimeError("Flower training finished without aggregated parameters.")

global_model = MLP(INPUT_DIM)
set_params(global_model, parameters_to_ndarrays(strategy.final_parameters))
final_global, prob_te, pred_te = evaluate(global_model, X_test_t, y_test_t)
print(f"\nFinal Federated Global on Test: {final_global}")


# ---------------------------------------------------------------------------
# 4. Centralized baseline (same MLP trained on union of client data)
# ---------------------------------------------------------------------------
print("\n--- Centralized baseline (same architecture, all client data pooled) ---")
X_all = torch.cat([x for x, _ in client_data], dim=0)
y_all = torch.cat([y for _, y in client_data], dim=0)
central = MLP(INPUT_DIM)
central = local_train(
    central,
    X_all,
    y_all,
    epochs=ROUNDS * LOCAL_EPOCHS // 3,
    lr=LR,
    bs=BS,
)
central_metrics, _, _ = evaluate(central, X_test_t, y_test_t)
print(f"Centralized Test: {central_metrics}")


# ---------------------------------------------------------------------------
# 5. Per-client local-only baselines (each client trains alone)
# ---------------------------------------------------------------------------
print("\n--- Local-only baselines (each client, no FL) ---")
local_only_metrics = []
for c, (Xc, yc) in enumerate(client_data):
    m_local = MLP(INPUT_DIM)
    m_local = local_train(
        m_local,
        Xc,
        yc,
        epochs=ROUNDS * LOCAL_EPOCHS // 3,
        lr=LR,
        bs=BS,
    )
    mm, _, _ = evaluate(m_local, X_test_t, y_test_t)
    print(f"Client {c} local-only test: acc={mm['accuracy']:.4f}  AUC={mm['roc_auc']:.4f}")
    local_only_metrics.append(mm)


# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------
rounds = history["round"]
acc = [m["accuracy"] for m in history["global_test"]]
auc = [m["roc_auc"] for m in history["global_test"]]
loss = [m["loss"] for m in history["global_test"]]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].plot(rounds, acc, marker="o", color="#1f77b4")
axes[0].set_title("Global Test Accuracy")
axes[1].plot(rounds, auc, marker="o", color="#2ca02c")
axes[1].set_title("Global Test ROC-AUC")
axes[2].plot(rounds, loss, marker="o", color="#d62728")
axes[2].set_title("Global Test Loss")
for ax in axes:
    ax.set_xlabel("Federated Round")
    ax.grid(alpha=0.3)
plt.suptitle("Flower FedAvg Convergence - XAUUSD Directional Forecasting", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/fl_convergence.png", dpi=160, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 4.5))
for c in range(N_CLIENTS):
    accs = [m["accuracy"] for m in history["client_local"][c]]
    client_rounds = rounds[1 : 1 + len(accs)]
    plt.plot(client_rounds, accs, marker=".", label=f"Client {c} (after local update)")
plt.plot(rounds, acc, marker="o", color="black", linewidth=2, label="Global model")
plt.xlabel("Round")
plt.ylabel("Accuracy on Global Test")
plt.title("Per-Client vs Global Accuracy across Rounds")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG}/per_client_accuracy.png", dpi=160, bbox_inches="tight")
plt.close()

labels = ["Centralized", "Federated\n(FedAvg)", "Local C0", "Local C1", "Local C2"]
accs = [central_metrics["accuracy"], final_global["accuracy"]] + [
    m["accuracy"] for m in local_only_metrics
]
aucs = [central_metrics["roc_auc"], final_global["roc_auc"]] + [
    m["roc_auc"] for m in local_only_metrics
]

x = np.arange(len(labels))
w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar(x - w / 2, accs, w, label="Accuracy", color="#1f77b4")
b2 = ax.bar(x + w / 2, aucs, w, label="ROC-AUC", color="#ff7f0e")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.4, max(max(accs), max(aucs)) + 0.05)
ax.set_title("Centralized vs Federated vs Local-only - Test Performance")
ax.legend()
ax.grid(alpha=0.3, axis="y")
for bars in (b1, b2):
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center",
            fontsize=8,
        )
plt.tight_layout()
plt.savefig(f"{FIG}/comparison_bar.png", dpi=160, bbox_inches="tight")
plt.close()

cm = confusion_matrix(y_test_t.numpy().astype(int), pred_te)
fig, ax = plt.subplots(figsize=(4.2, 3.8))
im = ax.imshow(cm, cmap="Blues")
for (i, j), value in np.ndenumerate(cm):
    ax.text(
        j,
        i,
        str(value),
        ha="center",
        va="center",
        color="white" if value > cm.max() / 2 else "black",
        fontsize=12,
    )
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Down", "Up"])
ax.set_yticklabels(["Down", "Up"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - Federated Global Model (Test)")
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(f"{FIG}/cm_federated.png", dpi=160, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# 7. Save metrics
# ---------------------------------------------------------------------------
out = {
    "config": {
        "rounds": ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "lr": LR,
        "batch_size": BS,
        "n_clients": N_CLIENTS,
        "input_dim": INPUT_DIM,
        "model": "MLP(64-32-1)",
        "framework": "Flower FedAvg",
    },
    "client_periods": [
        {
            "client": i,
            "start": str(cdf["date"].iloc[0].date()),
            "end": str(cdf["date"].iloc[-1].date()),
            "n": int(len(cdf)),
            "up_rate": float(cdf["target"].mean()),
        }
        for i, cdf in enumerate(client_dfs)
    ],
    "federated_global_test": final_global,
    "centralized_test": central_metrics,
    "local_only_test": local_only_metrics,
    "convergence": {
        "rounds": rounds,
        "global_test": history["global_test"],
        "client_local": history["client_local"],
    },
}
with open(f"{RES}/cse711_metrics.json", "w", encoding="utf-8") as fjson:
    json.dump(out, fjson, indent=2, default=float)

rows = [
    {"setting": "Centralized", **central_metrics},
    {"setting": "Federated (FedAvg)", **final_global},
    {"setting": "Local-only Client 0", **local_only_metrics[0]},
    {"setting": "Local-only Client 1", **local_only_metrics[1]},
    {"setting": "Local-only Client 2", **local_only_metrics[2]},
]
pd.DataFrame(rows).to_csv(f"{RES}/cse711_comparison.csv", index=False)
print(f"\nAll FL outputs saved under {FIG}/ and {RES}/")
