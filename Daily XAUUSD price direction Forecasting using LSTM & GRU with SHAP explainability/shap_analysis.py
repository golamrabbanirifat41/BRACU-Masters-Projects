"""
CSE710: SHAP analysis for the best recurrent model (LSTM or GRU).
Produces flattened time-feature SHAP plots plus feature importance aggregated
across the lookback window.
"""
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import shap
except ImportError as exc:
    raise SystemExit(
        "The 'shap' package is required for shap_analysis.py. Install it before running this script."
    ) from exc


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE_DIR, "results")
FIG = os.path.join(BASE_DIR, "figures", "cse710")
os.makedirs(FIG, exist_ok=True)

bundle = joblib.load(os.path.join(RES, "cse710_best_model.pkl"))
model = tf.keras.models.load_model(bundle["model_path"])
feature_cols = bundle["feature_cols"]
lookback = bundle["lookback"]
X_test = bundle["X_test"]
y_test = bundle["y_test"]
X_background = bundle.get("X_background", X_test[: min(len(X_test), 64)])
best_name = bundle["best_name"]
print(f"Loaded best model: {best_name}")

sample_count = min(len(X_test), 128)
background_count = min(len(X_background), 64)
X_sample = X_test[:sample_count]
background = X_background[:background_count]

explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(X_sample)
if isinstance(shap_values, list):
    shap_values = shap_values[0]
shap_values = np.asarray(shap_values)
shap_values = np.squeeze(shap_values)
if shap_values.ndim != 3:
    raise ValueError(f"Expected SHAP values with 3 dimensions, got shape {shap_values.shape}")
print(f"SHAP values shape: {shap_values.shape}")

time_feature_names = []
for step in range(lookback):
    lag = lookback - step - 1
    for feat in feature_cols:
        suffix = "t" if lag == 0 else f"t-{lag}"
        time_feature_names.append(f"{feat} [{suffix}]")

X_flat = X_sample.reshape(sample_count, lookback * len(feature_cols))
shap_flat = shap_values.reshape(sample_count, lookback * len(feature_cols))
X_flat_df = pd.DataFrame(X_flat, columns=time_feature_names)

plt.figure()
shap.summary_plot(
    shap_flat,
    X_flat_df,
    feature_names=time_feature_names,
    show=False,
    max_display=15,
)
plt.title(f"SHAP Summary (Beeswarm) - {best_name} on XAUUSD", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "shap_beeswarm.png"), dpi=160, bbox_inches="tight")
plt.close()

plt.figure()
shap.summary_plot(
    shap_flat,
    X_flat_df,
    feature_names=time_feature_names,
    plot_type="bar",
    show=False,
    max_display=15,
)
plt.title(f"Mean |SHAP| Time-Feature Importance - {best_name}", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "shap_bar.png"), dpi=160, bbox_inches="tight")
plt.close()

importance = pd.DataFrame(
    {
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_values).mean(axis=(0, 1)),
    }
).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
importance.to_csv(os.path.join(RES, "cse710_shap_importance.csv"), index=False)
print("\nTop 10 features by mean |SHAP| aggregated across timesteps:")
print(importance.head(10).to_string(index=False))

top4 = importance["feature"].head(4).tolist()
for feat in top4:
    feat_idx = feature_cols.index(feat)
    newest_step = lookback - 1
    flat_idx = newest_step * len(feature_cols) + feat_idx
    plt.figure()
    shap.dependence_plot(
        flat_idx,
        shap_flat,
        X_flat_df,
        feature_names=time_feature_names,
        show=False,
    )
    plt.title(f"SHAP Dependence - {feat} [t]", fontsize=11)
    plt.tight_layout()
    safe = feat.replace("/", "_")
    plt.savefig(os.path.join(FIG, f"shap_dependence_{safe}.png"), dpi=160, bbox_inches="tight")
    plt.close()

# Pick the waterfall example from the same subset used for SHAP generation.
y_sample = y_test[:sample_count]
prob = model.predict(X_sample, verbose=0).ravel()
pred = (prob >= 0.5).astype(int)
correct_up = np.where((pred == 1) & (y_sample == 1))[0]
idx = int(correct_up[len(correct_up) // 2]) if len(correct_up) else 0

expected_value = model(background, training=False).numpy().mean()
explanation = shap.Explanation(
    values=shap_flat[idx],
    base_values=float(expected_value),
    data=X_flat_df.iloc[idx].values,
    feature_names=time_feature_names,
)
plt.figure()
shap.plots.waterfall(explanation, max_display=12, show=False)
plt.title(f"SHAP Waterfall - Sample #{idx} (Predicted UP, Actual UP)", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "shap_waterfall.png"), dpi=160, bbox_inches="tight")
plt.close()

print(f"\nAll SHAP figures saved under {FIG}/")
