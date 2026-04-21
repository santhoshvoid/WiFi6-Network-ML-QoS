import joblib
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 1. Feature columns (same order!)
# -------------------------------
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

joblib.dump(FEATURE_COLS, 'feature_cols.pkl')
print("✅ feature_cols.pkl saved")


# -------------------------------
# 2. Label Encoder
# -------------------------------
traffic_types = ['HTTP', 'IoT', 'VPN', 'Video', 'VoIP']

le = LabelEncoder()
le.fit(traffic_types)

joblib.dump(le, 'label_encoder.pkl')
print("✅ label_encoder.pkl saved")

print("\nMapping:")
for t in traffic_types:
    print(f"{t} → {le.transform([t])[0]}")