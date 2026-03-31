"""
SRIM EXYZ plotter — three plots for PBW report
===============================================
Outlier ions (large-angle nuclear scattering events) removed
per depth bin using IQR clipping (k=3) before computing straggle.

Plot 1: Ion trajectories — spread through 1 µm PMMA
Plot 2: Range distribution — ion exit depth histogram
Plot 3: Lateral straggle σ(z) vs depth (raw vs cleaned overlay)

Run:  python srim_exyz_plots.py
"""

import numpy as np
import matplotlib.pyplot as plt

EXYZ_FILE = "/home/devinaak/Documents/FYP_docs/data/SRIM/EXYZ.txt"
BG = '#f8f8f6'
B, T, A, R, GR = '#185FA5', '#0F6E56', '#BA7517', '#A32D2D', '#888780'


def iqr_mask(arr, k=3.0):
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    return (arr >= q1 - k * iqr) & (arr <= q3 + k * iqr)


print("Parsing EXYZ.txt ...")

ions = {}
with open(EXYZ_FILE, 'r') as f:
    for line in f:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            iid  = int(parts[0])
            x_nm = float(parts[2]) / 10
            y_nm = float(parts[3]) / 10
            z_nm = float(parts[4]) / 10
        except ValueError:
            continue
        if iid not in ions:
            ions[iid] = []
        ions[iid].append((x_nm, y_nm, z_nm))

print(f"  Loaded {len(ions)} ion trajectories")

final_x = np.array([ions[i][-1][0] for i in ions])
final_y = np.array([ions[i][-1][1] for i in ions])
final_z = np.array([ions[i][-1][2] for i in ions])
final_r = np.sqrt(final_y**2 + final_z**2)

all_x, all_y, all_z = [], [], []
for traj in ions.values():
    for (x, y, z) in traj:
        all_x.append(x); all_y.append(y); all_z.append(z)
all_x = np.array(all_x)
all_y = np.array(all_y)
all_z = np.array(all_z)
all_r = np.sqrt(all_y**2 + all_z**2)


# ── PLOT 1 : trajectories ─────────────────────────────────────────────────────
print("Generating Plot 1: ion trajectories ...")
N_SHOW = 20
sample_ids = list(ions.keys())[:N_SHOW]

fig1, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=BG)
for ax in axes: ax.set_facecolor(BG)

ax = axes[0]
for iid in sample_ids:
    traj = np.array(ions[iid])
    ax.plot(traj[:, 0], traj[:, 1], color=B, lw=0.7, alpha=0.55)
ax.axvline(1000, color=R, lw=1.2, ls='--', label='1 µm depth')
ax.set_xlabel('Depth in PMMA (nm)', fontsize=11)
ax.set_ylabel('Lateral Y (nm)', fontsize=11)
ax.set_title('Ion trajectories — side view (X–Y)', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, lw=0.3, alpha=0.35); ax.tick_params(labelsize=9)

mask_exit = iqr_mask(final_r)
n_out = (~mask_exit).sum()
ax = axes[1]
ax.scatter(final_y[mask_exit],  final_z[mask_exit],
           s=8, color=T, alpha=0.5, label='typical ions')
ax.scatter(final_y[~mask_exit], final_z[~mask_exit],
           s=22, color=R, alpha=0.85, marker='x',
           label=f'nuclear scatter outliers (n={n_out})')
theta = np.linspace(0, 2*np.pi, 200)
for mult, ls in [(1, '-'), (2, '--')]:
    sr = np.std(final_r[mask_exit]) * mult
    ax.plot(sr*np.cos(theta), sr*np.sin(theta), color=A, lw=1.2, ls=ls,
            label=f'{mult}σ = {sr:.1f} nm')
ax.set_xlabel('Lateral Y (nm)', fontsize=11); ax.set_ylabel('Lateral Z (nm)', fontsize=11)
ax.set_title('Exit spread — end view (Y–Z)', fontsize=11)
ax.set_aspect('equal'); ax.legend(fontsize=8)
ax.grid(True, lw=0.3, alpha=0.35); ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig('srim_trajectories.png', dpi=150, bbox_inches='tight')
print("  Saved: srim_trajectories.png"); plt.close()


# ── PLOT 2 : range distribution ───────────────────────────────────────────────
print("Generating Plot 2: range distribution ...")
mask_range = iqr_mask(final_x)
x_clean    = final_x[mask_range]
mean_x, std_x = np.mean(x_clean), np.std(x_clean)

fig2, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
ax.set_facecolor(BG)
ax.hist(x_clean, bins=50, color=B, alpha=0.8, edgecolor='white', lw=0.3)
ax.axvline(mean_x, color=A, lw=1.8, ls='--', label=f'Mean = {mean_x:.0f} nm')
ax.axvline(mean_x - std_x, color=A, lw=1, ls=':')
ax.axvline(mean_x + std_x, color=A, lw=1, ls=':', label=f'±1σ = {std_x:.0f} nm')
ax.set_xlabel('Depth in PMMA (nm)', fontsize=12); ax.set_ylabel('Ion count', fontsize=12)
ax.set_title('Ion range distribution — 2 MeV H⁺ in PMMA (1 µm)', fontsize=12)
ax.legend(fontsize=10); ax.grid(True, lw=0.4, alpha=0.4); ax.tick_params(labelsize=10)
print(f"  Mean: {mean_x:.1f} nm  σ = {std_x:.1f} nm  (removed {(~mask_range).sum()} outliers)")
plt.tight_layout()
plt.savefig('srim_range_distribution.png', dpi=150, bbox_inches='tight')
print("  Saved: srim_range_distribution.png"); plt.close()


# ── PLOT 3 : straggle vs depth ────────────────────────────────────────────────
print("Generating Plot 3: lateral straggle vs depth ...")

n_bins = 60
depth_edges = np.linspace(0, 1000, n_bins + 1)
bin_centres  = 0.5 * (depth_edges[:-1] + depth_edges[1:])
sig_raw, sig_clean, n_removed = [], [], []

for i in range(n_bins):
    mask_bin = (all_x >= depth_edges[i]) & (all_x < depth_edges[i+1])
    r_bin = all_r[mask_bin]
    if len(r_bin) < 10:
        sig_raw.append(np.nan); sig_clean.append(np.nan); n_removed.append(0)
        continue
    sig_raw.append(np.std(r_bin))
    good = iqr_mask(r_bin)
    sig_clean.append(np.std(r_bin[good]))
    n_removed.append((~good).sum())

sig_raw   = np.array(sig_raw)
sig_clean = np.array(sig_clean)

fig3, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
ax.set_facecolor(BG)
ax.plot(bin_centres, sig_raw,   color=GR, lw=1.5, ls='--', alpha=0.7,
        label='raw (incl. nuclear scatter outliers)')
ax.plot(bin_centres, sig_clean, color=T,  lw=2.2,
        label='outliers removed (IQR ×3)')
ax.axhline(3, color=R, lw=1.5, ls=':', label='3 nm target')
ax.set_xlabel('Depth in PMMA (nm)', fontsize=12)
ax.set_ylabel('Lateral straggle σ_r (nm)', fontsize=12)
ax.set_title('Lateral straggle vs depth — 2 MeV H⁺ in PMMA', fontsize=12)
ax.legend(fontsize=10); ax.grid(True, lw=0.4, alpha=0.4)
ax.set_xlim(0, 1000); ax.tick_params(labelsize=10)

exit_raw   = float(np.nanmean(sig_raw[-5:]))
exit_clean = float(np.nanmean(sig_clean[-5:]))
print(f"  σ_radial at exit — raw: {exit_raw:.3f} nm  |  cleaned: {exit_clean:.3f} nm")
print(f"  Total outlier points removed: {sum(n_removed)}")
plt.tight_layout()
plt.savefig('/home/devinaak/Documents/FYP_docs/notes/report/images/srim_lateral_straggle.png', dpi=150, bbox_inches='tight')
print("  Saved: srim_lateral_straggle.png"); plt.close()

print("\nAll done.")
