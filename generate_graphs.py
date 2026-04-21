# =============================================================
# generate_graphs.py
#
# PURPOSE:
#   Generate all comparison graphs for the research paper
#   using the 4 CSV files produced by 4_wifi_qos_validate.cc
#
# INPUT FILES NEEDED:
#   results_without_ml.csv
#   results_with_ml.csv
#   stats_without_ml.csv
#   stats_with_ml.csv
#
# RUN:
#   python3 generate_graphs.py
#   (run on your Mac after copying CSVs from Ubuntu VM)
#
# OUTPUT:
#   graphs/ folder with 10 paper-ready PNG files
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

# ── Load CSVs ─────────────────────────────────────────────────
FILES = {
    'res_base':  'results_without_ml.csv',
    'res_ml':    'results_with_ml.csv',
    'stat_base': 'stats_without_ml.csv',
    'stat_ml':   'stats_with_ml.csv',
}

dfs = {}
for key, fname in FILES.items():
    if not os.path.exists(fname):
        print(f"❌  Missing: {fname}")
        print("    Copy all 4 CSV files from Ubuntu VM first.")
        sys.exit(1)
    dfs[key] = pd.read_csv(fname)
    print(f"✅  Loaded {fname}  ({len(dfs[key])} rows)")

res_base  = dfs['res_base']
res_ml    = dfs['res_ml']
stat_base = dfs['stat_base']
stat_ml   = dfs['stat_ml']

os.makedirs("graphs", exist_ok=True)

# ── Style ──────────────────────────────────────────────────────
C_BASE = '#ff6b6b'   # red  — without ML
C_ML   = '#51cf66'   # green — with ML
C_DARK = '#2d3436'
TRAFFIC_TYPES = ['VoIP', 'Video', 'HTTP', 'VPN', 'IoT']

plt.rcParams.update({
    'font.family':  'DejaVu Sans',
    'font.size':    11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi':   120,
})

# ── Helper: per-traffic-type average from stats CSV ───────────
def per_type_stat(stat_df, res_df, metric):
    """
    stat_df has flow_id and metric columns.
    res_df  has flow_id (0-indexed) and traffic_type.
    FlowMonitor flow IDs start at 1 so flow_id in stat = i+1.
    """
    # Build flow_id → traffic_type map from results CSV
    type_map = {}
    for _, row in res_df.iterrows():
        fid = int(row['flow_id']) + 1   # FlowMonitor starts at 1
        type_map[fid] = row['traffic_type']

    result = {t: [] for t in TRAFFIC_TYPES}
    for _, row in stat_df.iterrows():
        fid   = int(row['flow_id'])
        ttype = type_map.get(fid, None)
        if ttype and ttype in result:
            result[ttype].append(row[metric])

    return {t: np.mean(v) if v else 0 for t, v in result.items()}

# ── Helper: save figure ───────────────────────────────────────
def save(name):
    path = f"graphs/{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {path}")

# =============================================================
# GRAPH 1 — Overall Summary Bar Chart (4 metrics)
# =============================================================
print("\nGenerating graphs...")

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle(
    'ML vs No-ML: Overall QoS Summary\n'
    'Wi-Fi 6 (802.11ax) Network Simulation',
    fontsize=13, fontweight='bold', y=1.02)

metrics = [
    ('throughput_mbps', 'Avg Throughput (Mbps)', True),
    ('mean_delay_ms',   'Avg Delay (ms)',        False),
    ('mean_jitter_ms',  'Avg Jitter (ms)',       False),
    ('packet_loss_rate','Avg Packet Loss (%)',   False),
]

for ax, (col, label, higher_better) in zip(axes, metrics):
    v_base = stat_base[col].mean()
    v_ml   = stat_ml[col].mean()

    if col == 'packet_loss_rate':
        v_base *= 100
        v_ml   *= 100

    bars = ax.bar(['No ML', 'With ML'],
                  [v_base, v_ml],
                  color=[C_BASE, C_ML],
                  width=0.5,
                  edgecolor='white',
                  linewidth=1.2)

    # Improvement label
    if v_base != 0:
        pct = (v_ml - v_base) / abs(v_base) * 100
    else:
        pct = 0

    arrow = '↑' if pct > 0 else '↓'
    color = C_ML if (pct > 0) == higher_better else C_BASE
    ax.set_title(f"{label}\n{arrow}{abs(pct):.1f}%",
                 color=color, fontsize=10)
    ax.set_ylabel(label)
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
    ax.set_ylim(0, max(v_base, v_ml) * 1.3 + 0.001)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

save("01_overall_summary")

# =============================================================
# GRAPH 2 — Per Traffic Type: Delay
# =============================================================
delay_base = per_type_stat(stat_base, res_base, 'mean_delay_ms')
delay_ml   = per_type_stat(stat_ml,   res_ml,   'mean_delay_ms')

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(TRAFFIC_TYPES))
w = 0.35

b1 = ax.bar(x - w/2,
            [delay_base[t] for t in TRAFFIC_TYPES],
            w, label='Without ML', color=C_BASE,
            edgecolor='white')
b2 = ax.bar(x + w/2,
            [delay_ml[t] for t in TRAFFIC_TYPES],
            w, label='With ML', color=C_ML,
            edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(TRAFFIC_TYPES)
ax.set_ylabel('Mean End-to-End Delay (ms)')
ax.set_title('Delay per Traffic Type: ML vs No-ML\n'
             'Lower is better', fontweight='bold')
ax.legend()
ax.bar_label(b1, fmt='%.3f', padding=3, fontsize=8)
ax.bar_label(b2, fmt='%.3f', padding=3, fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add improvement % on top
for i, t in enumerate(TRAFFIC_TYPES):
    bv = delay_base[t]
    mv = delay_ml[t]
    if bv > 0:
        pct = (bv - mv) / bv * 100
        ax.text(i, max(bv, mv) + 0.02,
                f'{pct:+.1f}%',
                ha='center', va='bottom',
                fontsize=8, color=C_ML if pct > 0 else C_BASE)

save("02_delay_per_type")

# =============================================================
# GRAPH 3 — Per Traffic Type: Jitter
# =============================================================
jitter_base = per_type_stat(stat_base, res_base, 'mean_jitter_ms')
jitter_ml   = per_type_stat(stat_ml,   res_ml,   'mean_jitter_ms')

fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - w/2,
            [jitter_base[t] for t in TRAFFIC_TYPES],
            w, label='Without ML', color=C_BASE,
            edgecolor='white')
b2 = ax.bar(x + w/2,
            [jitter_ml[t] for t in TRAFFIC_TYPES],
            w, label='With ML', color=C_ML,
            edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(TRAFFIC_TYPES)
ax.set_ylabel('Mean Jitter (ms)')
ax.set_title('Jitter per Traffic Type: ML vs No-ML\n'
             'Lower is better', fontweight='bold')
ax.legend()
ax.bar_label(b1, fmt='%.3f', padding=3, fontsize=8)
ax.bar_label(b2, fmt='%.3f', padding=3, fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for i, t in enumerate(TRAFFIC_TYPES):
    bv = jitter_base[t]
    mv = jitter_ml[t]
    if bv > 0:
        pct = (bv - mv) / bv * 100
        ax.text(i, max(bv, mv) + 0.01,
                f'{pct:+.1f}%',
                ha='center', va='bottom',
                fontsize=8, color=C_ML if pct > 0 else C_BASE)

save("03_jitter_per_type")

# =============================================================
# GRAPH 4 — Per Traffic Type: Throughput
# =============================================================
tput_base = per_type_stat(stat_base, res_base, 'throughput_mbps')
tput_ml   = per_type_stat(stat_ml,   res_ml,   'throughput_mbps')

fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - w/2,
            [tput_base[t] for t in TRAFFIC_TYPES],
            w, label='Without ML', color=C_BASE,
            edgecolor='white')
b2 = ax.bar(x + w/2,
            [tput_ml[t] for t in TRAFFIC_TYPES],
            w, label='With ML', color=C_ML,
            edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(TRAFFIC_TYPES)
ax.set_ylabel('Throughput (Mbps)')
ax.set_title('Throughput per Traffic Type: ML vs No-ML\n'
             'Higher is better', fontweight='bold')
ax.legend()
ax.bar_label(b1, fmt='%.3f', padding=3, fontsize=8)
ax.bar_label(b2, fmt='%.3f', padding=3, fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

save("04_throughput_per_type")

# =============================================================
# GRAPH 5 — RU Allocation Comparison per Traffic Type
# =============================================================
ru_base = res_base.groupby('traffic_type')['ru'].mean()
ru_ml   = res_ml.groupby('traffic_type')['ru'].mean()

fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - w/2,
            [ru_base.get(t, 0) for t in TRAFFIC_TYPES],
            w, label='Without ML', color=C_BASE,
            edgecolor='white')
b2 = ax.bar(x + w/2,
            [ru_ml.get(t, 0) for t in TRAFFIC_TYPES],
            w, label='With ML', color=C_ML,
            edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(TRAFFIC_TYPES)
ax.set_ylabel('OFDMA Resource Units (RU)')
ax.set_title('ML-Driven OFDMA RU Allocation per Traffic Type\n'
             'ML assigns different RUs based on traffic needs',
             fontweight='bold')
ax.legend()
ax.bar_label(b1, fmt='%.0f', padding=3, fontsize=9)
ax.bar_label(b2, fmt='%.0f', padding=3, fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

save("05_ru_allocation")

# =============================================================
# GRAPH 6 — TWT Interval per Traffic Type
# =============================================================
twt_base = res_base.groupby('traffic_type')['twt_ms'].mean()
twt_ml   = res_ml.groupby('traffic_type')['twt_ms'].mean()

fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - w/2,
            [twt_base.get(t, 0) for t in TRAFFIC_TYPES],
            w, label='Without ML', color=C_BASE,
            edgecolor='white')
b2 = ax.bar(x + w/2,
            [twt_ml.get(t, 0) for t in TRAFFIC_TYPES],
            w, label='With ML', color=C_ML,
            edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(TRAFFIC_TYPES)
ax.set_ylabel('TWT Interval (ms)')
ax.set_title('ML-Driven TWT Interval per Traffic Type\n'
             'ML assigns shorter TWT to latency-sensitive traffic',
             fontweight='bold')
ax.legend()
ax.bar_label(b1, fmt='%.0f ms', padding=3, fontsize=8)
ax.bar_label(b2, fmt='%.0f ms', padding=3, fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

save("06_twt_interval")

# =============================================================
# GRAPH 7 — Priority Distribution (Pie Charts)
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Priority Assignment: ML vs No-ML',
             fontsize=13, fontweight='bold')

pie_colors = ['#ff6b6b', '#ffd43b', '#51cf66', '#339af0']

for ax, df, title in zip(
    axes,
    [res_base, res_ml],
    ['Without ML', 'With ML']
):
    counts = df['priority_name'].value_counts()
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct='%1.1f%%',
        colors=pie_colors[:len(counts)],
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title(title, fontsize=11)

save("07_priority_distribution")

# =============================================================
# GRAPH 8 — Bandwidth Allocation per Traffic Type
# =============================================================
bw_base = res_base.groupby('traffic_type')['bandwidth_mhz'].mean()
bw_ml   = res_ml.groupby('traffic_type')['bandwidth_mhz'].mean()

fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - w/2,
            [bw_base.get(t, 0) for t in TRAFFIC_TYPES],
            w, label='Without ML', color=C_BASE,
            edgecolor='white')
b2 = ax.bar(x + w/2,
            [bw_ml.get(t, 0) for t in TRAFFIC_TYPES],
            w, label='With ML', color=C_ML,
            edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(TRAFFIC_TYPES)
ax.set_ylabel('Allocated Bandwidth (MHz)')
ax.set_title('Bandwidth Allocation per Traffic Type\n'
             'ML differentiates based on traffic requirements',
             fontweight='bold')
ax.legend()
ax.bar_label(b1, fmt='%.0f MHz', padding=3, fontsize=8)
ax.bar_label(b2, fmt='%.0f MHz', padding=3, fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

save("08_bandwidth_allocation")

# =============================================================
# GRAPH 9 — Improvement Summary (Horizontal Bar)
# =============================================================
fig, ax = plt.subplots(figsize=(10, 6))

improvements = {}
for t in TRAFFIC_TYPES:
    db = delay_base[t]
    dm = delay_ml[t]
    jb = jitter_base[t]
    jm = jitter_ml[t]
    improvements[t] = {
        'Delay':  (db - dm) / db * 100 if db > 0 else 0,
        'Jitter': (jb - jm) / jb * 100 if jb > 0 else 0,
    }

labels_bar = []
values_bar = []
colors_bar = []

for t in TRAFFIC_TYPES:
    for metric in ['Delay', 'Jitter']:
        pct = improvements[t][metric]
        labels_bar.append(f"{t} — {metric}")
        values_bar.append(pct)
        colors_bar.append(C_ML if pct >= 0 else C_BASE)

y_pos = np.arange(len(labels_bar))
ax.barh(y_pos, values_bar, color=colors_bar,
        edgecolor='white', height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_bar, fontsize=9)
ax.axvline(x=0, color=C_DARK, linewidth=0.8, linestyle='--')
ax.set_xlabel('Improvement (%) — Positive = ML is better')
ax.set_title('QoS Improvement with ML vs No-ML\n'
             'Delay and Jitter reduction per traffic type',
             fontweight='bold')

for i, (v, c) in enumerate(zip(values_bar, colors_bar)):
    ax.text(v + (1 if v >= 0 else -1),
            i, f'{v:+.1f}%',
            va='center', ha='left' if v >= 0 else 'right',
            fontsize=8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

save("09_improvement_summary")

# =============================================================
# GRAPH 10 — Combined QoS Dashboard (2x2)
# =============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'ML-Driven QoS Optimization in Wi-Fi 6 (802.11ax)\n'
    'Complete Performance Dashboard',
    fontsize=14, fontweight='bold', y=1.01)

# Top-left: Delay per type
ax = axes[0][0]
b1 = ax.bar(x - w/2,
            [delay_base[t] for t in TRAFFIC_TYPES],
            w, label='No ML', color=C_BASE, edgecolor='white')
b2 = ax.bar(x + w/2,
            [delay_ml[t] for t in TRAFFIC_TYPES],
            w, label='ML', color=C_ML, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(TRAFFIC_TYPES, fontsize=9)
ax.set_ylabel('Delay (ms)')
ax.set_title('End-to-End Delay', fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Top-right: Jitter per type
ax = axes[0][1]
b1 = ax.bar(x - w/2,
            [jitter_base[t] for t in TRAFFIC_TYPES],
            w, label='No ML', color=C_BASE, edgecolor='white')
b2 = ax.bar(x + w/2,
            [jitter_ml[t] for t in TRAFFIC_TYPES],
            w, label='ML', color=C_ML, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(TRAFFIC_TYPES, fontsize=9)
ax.set_ylabel('Jitter (ms)')
ax.set_title('Jitter', fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Bottom-left: RU allocation
ax = axes[1][0]
b1 = ax.bar(x - w/2,
            [ru_base.get(t, 0) for t in TRAFFIC_TYPES],
            w, label='No ML', color=C_BASE, edgecolor='white')
b2 = ax.bar(x + w/2,
            [ru_ml.get(t, 0) for t in TRAFFIC_TYPES],
            w, label='ML', color=C_ML, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(TRAFFIC_TYPES, fontsize=9)
ax.set_ylabel('Resource Units (RU)')
ax.set_title('OFDMA RU Allocation', fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Bottom-right: TWT intervals
ax = axes[1][1]
b1 = ax.bar(x - w/2,
            [twt_base.get(t, 0) for t in TRAFFIC_TYPES],
            w, label='No ML', color=C_BASE, edgecolor='white')
b2 = ax.bar(x + w/2,
            [twt_ml.get(t, 0) for t in TRAFFIC_TYPES],
            w, label='ML', color=C_ML, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(TRAFFIC_TYPES, fontsize=9)
ax.set_ylabel('TWT Interval (ms)')
ax.set_title('TWT Scheduling Interval', fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

save("10_dashboard")

# =============================================================
# Print Summary Table
# =============================================================
print("\n" + "="*60)
print("  RESULTS SUMMARY — ML vs No-ML")
print("="*60)
print(f"{'Metric':<30} {'No ML':>10} {'With ML':>10} {'Change':>10}")
print("-"*60)

summary_metrics = [
    ('Avg Throughput (Mbps)',
     stat_base['throughput_mbps'].mean(),
     stat_ml['throughput_mbps'].mean(), True),
    ('Avg Delay (ms)',
     stat_base['mean_delay_ms'].mean(),
     stat_ml['mean_delay_ms'].mean(), False),
    ('Avg Jitter (ms)',
     stat_base['mean_jitter_ms'].mean(),
     stat_ml['mean_jitter_ms'].mean(), False),
    ('Avg Packet Loss (%)',
     stat_base['packet_loss_rate'].mean()*100,
     stat_ml['packet_loss_rate'].mean()*100, False),
]

for name, base_v, ml_v, higher_better in summary_metrics:
    if base_v != 0:
        pct = (ml_v - base_v) / abs(base_v) * 100
    else:
        pct = 0
    better = (pct > 0) == higher_better
    arrow  = '✅' if better else '⚠️'
    print(f"{name:<30} {base_v:>10.4f} {ml_v:>10.4f}"
          f" {arrow}{pct:>+7.1f}%")

print("="*60)

print("\n--- Per Traffic Type: Delay ---")
print(f"{'Type':<8} {'No ML (ms)':>12} {'With ML (ms)':>14} {'Improvement':>12}")
print("-"*50)
for t in TRAFFIC_TYPES:
    bv = delay_base[t]
    mv = delay_ml[t]
    pct = (bv - mv) / bv * 100 if bv > 0 else 0
    print(f"{t:<8} {bv:>12.4f} {mv:>14.4f} {pct:>+11.1f}%")

print("\n--- Per Traffic Type: Jitter ---")
print(f"{'Type':<8} {'No ML (ms)':>12} {'With ML (ms)':>14} {'Improvement':>12}")
print("-"*50)
for t in TRAFFIC_TYPES:
    bv = jitter_base[t]
    mv = jitter_ml[t]
    pct = (bv - mv) / bv * 100 if bv > 0 else 0
    print(f"{t:<8} {bv:>12.4f} {mv:>14.4f} {pct:>+11.1f}%")

print(f"\n✅  All graphs saved in ./graphs/")
print("    10 graphs ready for your research paper\n")
