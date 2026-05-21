"""
CSE710: Train recurrent baselines (LSTM and GRU) for XAUUSD direction.
Uses a chronological split and rolling lookback windows to model the feature
sequence leading into each prediction date.
"""
import json
import os
import random

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

#DATA LOAD AND FEATURE SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE_DIR, "xauusd_features.csv")
FIG = os.path.join(BASE_DIR, "figures", "cse710")
RES = os.path.join(BASE_DIR, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

LOOKBACK = 20
BATCH_SIZE = 32
EPOCHS = 40
PATIENCE = 6
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def make_sequences(X, y, dates, lookback):
    sequences, labels, sample_dates, end_idx = [], [], [], []
    for end in range(lookback - 1, len(X)):
        start = end - lookback + 1
        sequences.append(X[start:end + 1])
        labels.append(y[end])
        sample_dates.append(dates[end])
        end_idx.append(end)
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(sample_dates),
        np.asarray(end_idx, dtype=np.int32),
    )

#computes classification marices returns accuracy, precision, recall, f1, etc.
def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

#build sequential rnn model for lstm and gru 
def build_model(kind, lookback, n_features):
    rnn_layer = tf.keras.layers.LSTM if kind == "LSTM" else tf.keras.layers.GRU
    inputs = tf.keras.Input(shape=(lookback, n_features), name="sequence")
    x = rnn_layer(64, dropout=0.2, recurrent_dropout=0.0, name=f"{kind.lower()}_layer")(inputs)
    x = tf.keras.layers.Dense(32, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_1")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="probability")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"xauusd_{kind.lower()}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model

#generate confusion matrix 
def plot_confusion_matrix(cm, name):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(
            j,
            i,
            str(v),
            ha="center",
            va="center",
            color="black" if v > cm.max() / 2 else "darkblue",
            fontsize=12,
        )
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Down", "Up"])
    ax.set_yticklabels(["Down", "Up"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {name} (Test)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"cm_{name}.png"), dpi=300)
    plt.close()


def plot_history(history, name):
    epochs = range(1, len(history.history["loss"]) + 1)
    plt.figure(figsize=(5.6, 4.2))
    plt.plot(epochs, history.history["loss"], label="Train Loss", linewidth=2)
    plt.plot(epochs, history.history["val_loss"], label="Val Loss", linewidth=2)
    if "auc" in history.history and "val_auc" in history.history:
        plt.plot(epochs, history.history["auc"], label="Train AUC", linewidth=1.5, linestyle="--")
        plt.plot(epochs, history.history["val_auc"], label="Val AUC", linewidth=1.5, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title(f"Training History - {name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"history_{name}.png"), dpi=160)
    plt.close()


df = pd.read_csv(DATA, parse_dates=["date"])
feature_cols = [c for c in df.columns if c not in ("date", "target")]
X = df[feature_cols].to_numpy(dtype=np.float32)
y = df["target"].to_numpy(dtype=np.int32)
dates = df["date"].to_numpy()

#scaling and spliting 
n = len(df)
i_tr = int(0.70 * n)
i_va = int(0.85 * n)

scaler = StandardScaler().fit(X[:i_tr])
X_scaled = scaler.transform(X).astype(np.float32)

X_seq, y_seq, seq_dates, end_idx = make_sequences(X_scaled, y, dates, LOOKBACK)
train_mask = end_idx < i_tr
val_mask = (end_idx >= i_tr) & (end_idx < i_va)
test_mask = end_idx >= i_va

X_tr, y_tr = X_seq[train_mask], y_seq[train_mask]
X_va, y_va = X_seq[val_mask], y_seq[val_mask]
X_te, y_te = X_seq[test_mask], y_seq[test_mask]
dates_tr = seq_dates[train_mask]
dates_va = seq_dates[val_mask]
dates_te = seq_dates[test_mask]

print(f"Lookback window: {LOOKBACK}")
print(f"Train sequences: {len(X_tr)}  Val sequences: {len(X_va)}  Test sequences: {len(X_te)}")
print(f"Train dates: {pd.Timestamp(dates_tr[0]).date()} -> {pd.Timestamp(dates_tr[-1]).date()}")
print(f"Val   dates: {pd.Timestamp(dates_va[0]).date()} -> {pd.Timestamp(dates_va[-1]).date()}")
print(f"Test  dates: {pd.Timestamp(dates_te[0]).date()} -> {pd.Timestamp(dates_te[-1]).date()}")

models = {
    "LSTM": build_model("LSTM", LOOKBACK, len(feature_cols)),
    "GRU": build_model("GRU", LOOKBACK, len(feature_cols)),
}

results = {}
roc_data = {}
histories = {}
trained_models = {}

for name, model in models.items():
    print(f"\n=== {name} ===")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=PATIENCE,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )
    histories[name] = history.history
    trained_models[name] = model
    plot_history(history, name)

    for split_name, Xs, ys in [("val", X_va, y_va), ("test", X_te, y_te)]:
        prob = model.predict(Xs, verbose=0).ravel()
        pred = (prob >= 0.5).astype(int)
        metrics = evaluate(ys, pred, prob)
        results.setdefault(name, {})[split_name] = metrics
        print(f"  {split_name}: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    test_prob = model.predict(X_te, verbose=0).ravel()
    test_pred = (test_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_te, test_pred)
    plot_confusion_matrix(cm, name)
    fpr, tpr, _ = roc_curve(y_te, test_prob)
    roc_data[name] = (fpr, tpr, results[name]["test"]["roc_auc"])

plt.figure(figsize=(5.2, 4.6))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - XAUUSD Sequence Classifiers (Test)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "roc_all.png"), dpi=160)
plt.close()

with open(os.path.join(RES, "cse710_metrics.json"), "w", encoding="utf-8") as fjson:
    json.dump(results, fjson, indent=2)

rows = []
for name, splits in results.items():
    for split_name, metrics in splits.items():
        rows.append({"model": name, "split": split_name, **metrics})
pd.DataFrame(rows).to_csv(os.path.join(RES, "cse710_metrics.csv"), index=False)

best = max(results, key=lambda k: results[k]["val"]["roc_auc"])
print(f"\nBest model by validation AUC: {best}")

best_model = trained_models[best]
model_path = os.path.join(RES, "cse710_best_model.keras")
best_model.save(model_path)

joblib.dump(
    {
        "model_path": model_path,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "lookback": LOOKBACK,
        "X_test": X_te,
        "y_test": y_te,
        "test_dates": dates_te,
        "X_background": X_tr[-min(len(X_tr), 128):],
        "best_name": best,
        "results": results,
        "history": histories.get(best, {}),
    },
    os.path.join(RES, "cse710_best_model.pkl"),
)
print(f"Saved best model bundle -> {os.path.join(RES, 'cse710_best_model.pkl')}")
