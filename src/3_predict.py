# =============================================================
# 3_predict.py
#
# PURPOSE:
#   Called at runtime by 4_wifi_qos_validate.cc via popen().
#   Loads the ns-3-trained model and returns QoS decisions.
#
# USAGE (by ns-3, not by you directly):
#   python3 3_predict.py <packet_size> <interval_ms> <num_stations>
#
# OUTPUT (space-separated on one line):
#   priority  ru  twt_ms  mcs
# =============================================================

import joblib
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Load model artifacts ──────────────────────────────────────
try:
    model     = joblib.load("model.pkl")
    feat_cols = joblib.load("feature_cols.pkl")
    le        = joblib.load("label_encoder.pkl")
except FileNotFoundError as e:
    # Safe defaults if model not found
    print(1, 10, 50, 6)
    sys.exit(0)

# ── Read inputs from ns-3 ─────────────────────────────────────
packet_size  = int(sys.argv[1])
interval_ms  = int(sys.argv[2])
num_stations = int(sys.argv[3]) if len(sys.argv) > 3 else 8

# ── Estimate ru and mcs for feature derivation ────────────────
# Use middle-range defaults for the derived features
# (actual values will be predicted by the model)
ru_est  = 10
mcs_est = 6

# ── Derived features (same logic as 2_train_model.py) ─────────
throughput_mbps = round(
    (packet_size * 8.0) / max(interval_ms, 1)
    * 1000.0 / 1e6, 6)

# Estimate delay/jitter from packet size and load
# These are approximations used only for prediction input
mean_delay_ms    = 0.30
mean_jitter_ms   = 0.20
packet_loss_rate = 0.00

# Infer traffic type encoding from packet size + interval
# Matches LabelEncoder alphabetical order:
# HTTP=0, IoT=1, VPN=2, Video=3, VoIP=4
if   packet_size <= 100:
    traffic_type_enc = 1   # IoT
elif packet_size <= 300:
    traffic_type_enc = 4   # VoIP
elif packet_size <= 850:
    traffic_type_enc = 0   # HTTP
elif packet_size <= 950:
    traffic_type_enc = 2   # VPN
else:
    traffic_type_enc = 3   # Video

throughput_efficiency = min(1.0,
    throughput_mbps / (ru_est / 37.0 * 143.4 + 1e-9))

channel_load_mbps = (
    num_stations * packet_size * 8.0
    / (interval_ms / 1000.0) / 1e6)

# ── Build feature vector in training column order ─────────────
feature_values = {
    'packet_size':           packet_size,
    'interval_ms':           interval_ms,
    'num_stations':          num_stations,
    'ru':                    ru_est,
    'mcs':                   mcs_est,
    'throughput_mbps':       throughput_mbps,
    'mean_delay_ms':         mean_delay_ms,
    'mean_jitter_ms':        mean_jitter_ms,
    'packet_loss_rate':      packet_loss_rate,
    'traffic_type_enc':      traffic_type_enc,
    'throughput_efficiency': throughput_efficiency,
    'channel_load_mbps':     channel_load_mbps,
}

X = np.array([[feature_values[c] for c in feat_cols]])

# ── Predict ───────────────────────────────────────────────────
pred = model.predict(X)[0]

# ── Clamp to valid Wi-Fi 6 ranges ─────────────────────────────
priority = int(np.clip(round(pred[0]), 0,   3))
ru       = int(np.clip(round(pred[1]), 1,  37))
twt      = int(np.clip(round(pred[2]), 5, 500))
mcs      = int(np.clip(round(pred[3]), 0,  11))

# ── Output — ns-3 reads exactly this line ─────────────────────
print(priority, ru, twt, mcs)
