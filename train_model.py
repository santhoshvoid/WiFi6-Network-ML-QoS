# =============================================================
# 2_train_model.py
#
# PURPOSE:
#   Train a multi-output Random Forest on the CSV produced
#   by 1_data_collector.cc.
#   Saves model.pkl and feature_cols.pkl for use by
#   3_predict.py (called at runtime by 4_wifi_qos_validate.cc)
#
# RUN:
#   python3 2_train_model.py
#   (run on your Mac after copying ns3_training_data.csv)
# =============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib
import os
import sys

DATA_FILE = "ns3_training_data.csv"

# ── Load data ─────────────────────────────────────────────────
if not os.path.exists(DATA_FILE):
    print(f"❌  {DATA_FILE} not found.")
    print("    Run 1_data_collector first:")
    print("    ./ns3 run scratch/1_data_collector")
    sys.exit(1)

df = pd.read_csv(DATA_FILE)
print(f"✅  Loaded {len(df)} rows from {DATA_FILE}")
print(f"\nTraffic type counts:\n{df['traffic_type'].value_counts()}")
print(f"\nRU range  : {df['ru'].min()} – {df['ru'].max()}")
print(f"MCS range : {df['mcs'].min()} – {df['mcs'].max()}")
print(f"nSta range: {df['num_stations'].min()} – {df['num_stations'].max()}")

# ── Encode traffic_type as integer ────────────────────────────
# LabelEncoder sorts alphabetically:
#   HTTP=0, IoT=1, VPN=2, Video=3, VoIP=4
le = LabelEncoder()
df['traffic_type_enc'] = le.fit_transform(df['traffic_type'])
joblib.dump(le, 'label_encoder.pkl')
print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── Derived features ──────────────────────────────────────────
# throughput efficiency: how well the RU is being used
df['throughput_efficiency'] = (
    df['throughput_mbps']
    / (df['ru'] / 37.0 * 143.4 + 1e-9)
).clip(0, 1)

# total channel load from all stations
df['channel_load_mbps'] = (
    df['num_stations']
    * df['packet_size'] * 8.0
    / (df['interval_ms'] / 1000.0)
    / 1e6
)

# ── Features and targets ──────────────────────────────────────
FEATURE_COLS = [
    'packet_size',
    'interval_ms',
    'num_stations',
    'ru',
    'mcs',
    'throughput_mbps',
    'mean_delay_ms',
    'mean_jitter_ms',
    'packet_loss_rate',
    'traffic_type_enc',
    'throughput_efficiency',
    'channel_load_mbps',
]

TARGET_COLS = [
    'priority',   # 0=Low  1=Medium  2=High(video)  3=High(VoIP)
    'ru',         # optimal RU count (1-37)
    'twt_ms',     # optimal TWT interval (ms)
    'mcs',        # optimal MCS index (0-11)
]

df = df.dropna(subset=FEATURE_COLS + TARGET_COLS)
print(f"\nClean rows: {len(df)}")

X = df[FEATURE_COLS]
y = df[TARGET_COLS]

# ── Train / test split (stratify on traffic type) ─────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=df['traffic_type_enc'])

print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")

# ── Train ─────────────────────────────────────────────────────
print("\nTraining Random Forest...")
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
print("Training complete ✅")

# ── Evaluate ──────────────────────────────────────────────────
y_pred = model.predict(X_test)

print("\n" + "="*50)
print("  Evaluation on ns-3 test data")
print("="*50)
for i, col in enumerate(TARGET_COLS):
    mae = mean_absolute_error(y_test[col], y_pred[:, i])
    r2  = r2_score(y_test[col],           y_pred[:, i])
    print(f"  {col:12s}  MAE = {mae:6.3f}   R² = {r2:.3f}")

# ── Feature importance ────────────────────────────────────────
imp = pd.DataFrame({
    'feature':    FEATURE_COLS,
    'importance': model.feature_importances_,
}).sort_values('importance', ascending=False)

print("\nFeature importance (top 6):")
print(imp.head(6).to_string(index=False))

# ── Save ──────────────────────────────────────────────────────
joblib.dump(model,        'model.pkl')
joblib.dump(FEATURE_COLS, 'feature_cols.pkl')
print("\n✅  model.pkl saved")
print("✅  feature_cols.pkl saved")

# ── Plots ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Predicted vs Actual — ns-3 Trained Model',
             fontsize=13, fontweight='bold')

for ax, col, i in zip(axes.flatten(), TARGET_COLS, range(4)):
    act  = y_test[col].values
    pred = y_pred[:, i]
    ax.scatter(act, pred, alpha=0.35, s=8, color='#339af0')
    mn, mx = min(act.min(), pred.min()), max(act.max(), pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=1.2,
            label='Perfect')
    ax.set_xlabel(f'Actual {col}')
    ax.set_ylabel(f'Predicted {col}')
    ax.set_title(f'{col}   R²={r2_score(act, pred):.3f}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150)

fig2, ax2 = plt.subplots(figsize=(10, 5))
imp.plot(kind='barh', x='feature', y='importance',
         ax=ax2, color='#51cf66', legend=False)
ax2.set_title('Feature Importance')
ax2.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)

print("✅  model_evaluation.png saved")
print("✅  feature_importance.png saved")
print("\n🎉  Model trained on ns-3 data and ready!")
print("    Copy model.pkl + feature_cols.pkl + label_encoder.pkl")
print("    to ~/ns-3-dev/ then run 4_wifi_qos_validate.cc")
