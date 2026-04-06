"""
AFM roughness profile processor
Usage:  python afm_process.py <file1> [file2 ...]
Output: PNG plots + printed roughness metrics (Rq, Ra, Rz) per profile
"""

import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.signal import savgol_filter


# ── helpers ────────────────────────────────────────────────────────────────

def parse_afm_file(path):
    """
    Parse the whitespace-separated multi-profile AFM export.
    Returns list of (x_nm, y_nm) arrays, one per profile.
    Header structure:
        Profile 1   Profile 2   ...
        x   y       x   y       ...
        [m] [m]     [m] [m]     ...
        <data rows>
    """
    lines = Path(path).read_text().splitlines()

    # count profiles from header line 0
    header = lines[0]
    n_profiles = len(re.findall(r'Profile\s+\d+', header))

    # data starts at line 3 (0-indexed)
    xs = [[] for _ in range(n_profiles)]
    ys = [[] for _ in range(n_profiles)]

    for line in lines[3:]:
        vals = line.split()
        if not vals:
            continue
        # each profile contributes 2 columns; some rows may be ragged
        for i in range(n_profiles):
            try:
                x = float(vals[i * 2])
                y = float(vals[i * 2 + 1])
                xs[i].append(x)
                ys[i].append(y)
            except (IndexError, ValueError):
                pass

    profiles = []
    for i in range(n_profiles):
        x = np.array(xs[i]) * 1e9   # m → nm
        y = np.array(ys[i]) * 1e9   # m → nm
        # level: subtract mean (removes tilt offset)
        y = y - np.mean(y)
        profiles.append((x, y))

    return profiles


def roughness_metrics(y):
    """Return Rq, Ra, Rz (all in nm)."""
    rq = float(np.sqrt(np.mean(y ** 2)))
    ra = float(np.mean(np.abs(y)))
    rz = float(np.max(y) - np.min(y))
    return rq, ra, rz


def nc_factor(qz, rq):
    """Névot-Croce factor."""
    return float(np.exp(-qz ** 2 * rq ** 2))


# ── plotting ────────────────────────────────────────────────────────────────

COLORS = ['#1D9E75', '#178BD4', '#7F77DD', '#EF9F27', '#D85A30']
THRESHOLD_RQ = 1.0   # sub-1 nm target

def plot_file(path, out_dir=None):
    path = Path(path)
    profiles = parse_afm_file(path)
    n = len(profiles)
    name = path.stem

    fig = plt.figure(figsize=(12, 4 + 2.5 * n), constrained_layout=True)
    fig.suptitle(f'AFM surface profiles — {name}', fontsize=13, fontweight='500')

    gs = gridspec.GridSpec(n + 1, 3, figure=fig, height_ratios=[2.5] * n + [1.8])

    metrics_all = []

    for i, (x, y) in enumerate(profiles):
        col = COLORS[i % len(COLORS)]
        rq, ra, rz = roughness_metrics(y)
        metrics_all.append((i + 1, rq, ra, rz))

        # ── profile plot
        ax_p = fig.add_subplot(gs[i, :2])
        ax_p.plot(x, y, color=col, lw=0.8, alpha=0.9, label='raw')
        # smoothed envelope
        if len(y) > 11:
            sm = savgol_filter(y, min(11, len(y) if len(y) % 2 else len(y) - 1), 3)
            ax_p.plot(x, sm, color=col, lw=1.8, alpha=0.5, ls='--', label='smoothed')
        ax_p.axhline(0, color='grey', lw=0.5, ls=':')
        ax_p.fill_between(x, y, alpha=0.08, color=col)
        ax_p.set_ylabel('Height (nm)', fontsize=9)
        ax_p.set_xlabel('Position (nm)', fontsize=9)
        ax_p.set_title(f'Profile {i + 1}', fontsize=10, fontweight='500', loc='left')
        ax_p.tick_params(labelsize=8)
        ax_p.legend(fontsize=8, frameon=False)

        # ── metrics card
        ax_m = fig.add_subplot(gs[i, 2])
        ax_m.axis('off')
        rq_col = '#1D9E75' if rq < THRESHOLD_RQ else '#BA7517'
        text = (
            f'Rq = {rq:.3f} nm\n'
            f'Ra = {ra:.3f} nm\n'
            f'Rz = {rz:.3f} nm\n\n'
            f'NC (q=0.5) = {nc_factor(0.5, rq):.3f}\n'
            f'NC (q=1.0) = {nc_factor(1.0, rq):.3f}\n\n'
            f'{"✓ < 1 nm" if rq < THRESHOLD_RQ else "△ > 1 nm"}'
        )
        ax_m.text(0.05, 0.95, text, transform=ax_m.transAxes,
                  fontsize=9, va='top', family='monospace',
                  color=rq_col,
                  bbox=dict(boxstyle='round,pad=0.6', facecolor='#f7f7f5', edgecolor='#ddd', lw=0.5))

    # ── summary bar chart (bottom row)
    ax_bar = fig.add_subplot(gs[n, :])
    pnames = [f'P{m[0]}' for m in metrics_all]
    rqs = [m[1] for m in metrics_all]
    bar_cols = [('#1D9E75' if r < THRESHOLD_RQ else '#BA7517') for r in rqs]
    bars = ax_bar.bar(pnames, rqs, color=bar_cols, width=0.4, edgecolor='white', linewidth=0.5)
    ax_bar.axhline(THRESHOLD_RQ, color='#D85A30', lw=1, ls='--', label='1 nm target')
    ax_bar.axhline(3.0, color='#888', lw=0.8, ls=':', label='3 nm limit')
    for bar, val in zip(bars, rqs):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax_bar.set_ylabel('Rq (nm)', fontsize=9)
    ax_bar.set_title('Rq summary', fontsize=10, fontweight='500', loc='left')
    ax_bar.tick_params(labelsize=8)
    ax_bar.legend(fontsize=8, frameon=False)
    ax_bar.set_ylim(0, max(max(rqs) * 1.3, 1.5))

    # ── save
    out = Path(out_dir) if out_dir else path.parent
    out_path = out / f'{name}_afm.png'
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved → {out_path}')

    # ── print table
    print(f'{"Profile":<10} {"Rq (nm)":<12} {"Ra (nm)":<12} {"Rz (nm)":<12} {"NC q=0.5":<12} {"<1nm?"}')
    print('-' * 65)
    for pidx, rq, ra, rz in metrics_all:
        nc = nc_factor(0.5, rq)
        flag = '✓' if rq < THRESHOLD_RQ else '△'
        print(f'P{pidx:<9} {rq:<12.3f} {ra:<12.3f} {rz:<12.3f} {nc:<12.3f} {flag}')

    return metrics_all


# ── main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    files = sys.argv[1:] if len(sys.argv) > 1 else []
    if not files:
        print('Usage: python afm_process.py <file1> [file2 ...]')
        sys.exit(1)

    all_metrics = {}
    for f in files:
        print(f'\n{"="*60}')
        print(f'Processing: {f}')
        print('='*60)
        try:
            m = plot_file(f, out_dir='.')
            all_metrics[Path(f).stem] = m
        except Exception as e:
            print(f'  ERROR: {e}')

    # ── combined summary across all files
    if len(files) > 1:
        fig, ax = plt.subplots(figsize=(max(8, len(files)*1.5), 4))
        labels, rq_vals, colors = [], [], []
        for fname, metrics in all_metrics.items():
            for pidx, rq, ra, rz in metrics:
                labels.append(f'{fname}\nP{pidx}')
                rq_vals.append(rq)
                colors.append('#1D9E75' if rq < THRESHOLD_RQ else '#BA7517')
        bars = ax.bar(range(len(labels)), rq_vals, color=colors, width=0.6, edgecolor='white')
        ax.axhline(THRESHOLD_RQ, color='#D85A30', lw=1, ls='--', label='1 nm target')
        ax.axhline(3.0, color='#888', lw=0.8, ls=':', label='3 nm limit')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')
        ax.set_ylabel('Rq (nm)', fontsize=10)
        ax.set_title('Rq across all samples', fontsize=11, fontweight='500')
        ax.legend(fontsize=9, frameon=False)
        for bar, val in zip(bars, rq_vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)
        ax.set_ylim(0, max(max(rq_vals)*1.3, 1.5))
        fig.tight_layout()
        fig.savefig('all_samples_rq_summary.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
        print('\nSaved → all_samples_rq_summary.png')