"""
AFM profile plotter — Gwyddion multi-profile export
=====================================================
Reads the tab-separated format exported by Gwyddion (multiple profiles
side by side, headers: Profile 1 / Profile 2 ..., x [m] / y [m] columns).
Converts metres to nanometres and plots each profile on a separate figure,
with the profile number in the title and Rq/Ra/Rmax annotated on the plot.

Run:
    python afm_profile.py
    → prompted for filename
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ── Ask for filename ──────────────────────────────────────────────────────────
filename = input("Enter AFM profile filename: ").strip()
if not os.path.exists(filename):
    print(f"File not found: {filename}")
    exit(1)

# ── Parse the Gwyddion multi-profile format ───────────────────────────────────
# Layout:  Profile 1   (blank)   Profile 2   (blank) ...
#          x    y                x    y
#          [m]  [m]              [m]  [m]
#          data ...
with open(filename, 'r') as f:
    lines = f.readlines()

# Count profiles from header row 0
header = lines[0].split()
profile_names = [w for w in header if w.startswith('Profile')]
n_profiles = len(profile_names)
print(f"Found {n_profiles} profile(s): {', '.join(profile_names)}")

# Skip 3 header rows, read numeric data
profiles = {i: {'x': [], 'y': []} for i in range(n_profiles)}
for line in lines[3:]:
    parts = line.split()
    for p in range(n_profiles):
        xi = p * 2
        yi = p * 2 + 1
        if xi >= len(parts) or yi >= len(parts):
            continue
        if parts[xi] == '-' or parts[yi] == '-':
            continue
        try:
            profiles[p]['x'].append(float(parts[xi]))
            profiles[p]['y'].append(float(parts[yi]))
        except ValueError:
            continue

# Convert metres → nanometres
for p in range(n_profiles):
    profiles[p]['x'] = np.array(profiles[p]['x']) * 1e9
    profiles[p]['y'] = np.array(profiles[p]['y']) * 1e9

# ── Compute roughness metrics ─────────────────────────────────────────────────
def roughness(y):
    rq   = float(np.sqrt(np.mean((y - np.mean(y)) ** 2)))
    ra   = float(np.mean(np.abs(y - np.mean(y))))
    rmax = float(np.max(y) - np.min(y))
    return rq, ra, rmax

# ── Plot — one figure per profile ────────────────────────────────────────────
colors = ['#185FA5', '#0F6E56', '#BA7517', '#A32D2D', '#7F77DD', '#D85A30']
base = os.path.basename(os.path.splitext(filename)[0])

print("\nRoughness summary:")

saved_files = []

for p in range(n_profiles):
    x = profiles[p]['x']
    y = profiles[p]['y']
    name = profile_names[p]   # e.g. "Profile 1"
    num  = name.split()[-1]   # e.g. "1"

    if len(x) == 0:
        print(f"  {name}: no data — skipped")
        continue

    rq, ra, rmax = roughness(y)
    print(f"  {name}:  Rq = {rq:.3f} nm   Ra = {ra:.3f} nm   Rmax = {rmax:.3f} nm")

    y_zeroed = y - np.mean(y)
    col = colors[p % len(colors)]

    fig, ax = plt.subplots(figsize=(9, 4), facecolor='#f8f8f6')
    ax.set_facecolor('#f8f8f6')

    ax.plot(x, y_zeroed, color=col, lw=1.4)
    ax.fill_between(x, y_zeroed, alpha=0.08, color=col)
    ax.axhline(0, color='#aaa', lw=0.6, ls='--')

    # ── Roughness annotation box ──────────────────────────────────────────────
    metrics_text = (
        f"Rq  = {rq:.3f} nm\n"
        f"Ra  = {ra:.3f} nm\n"
        f"Rmax = {rmax:.3f} nm"
    )
    ax.text(
        0.98, 0.97, metrics_text,
        transform=ax.transAxes,
        fontsize=9, va='top', ha='right',
        fontfamily='monospace',
        color='#333',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#ccc', alpha=0.85, lw=0.8)
    )

    ax.set_title(f"{base}  —  Profile {num}", fontsize=12, fontweight='500',
                 color='#1a1a1a', pad=10)
    ax.set_xlabel('Position (nm)', fontsize=11)
    ax.set_ylabel('Height (nm)',   fontsize=11)
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.tick_params(labelsize=10)

    plt.tight_layout()

    outname = f"{base}_profile{num}.png"
    fig.savefig(outname, dpi=150, bbox_inches='tight')
    saved_files.append(outname)
    plt.show()
    plt.close(fig)

print(f"\nSaved {len(saved_files)} figure(s):")
for f in saved_files:
    print(f"  {f}")