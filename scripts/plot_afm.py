"""
AFM profile plotter — Gwyddion multi-profile export
=====================================================
Reads the tab-separated format exported by Gwyddion (multiple profiles
side by side, headers: Profile 1 / Profile 2 ..., x [m] / y [m] columns).
Converts metres to nanometres and plots each profile on one figure.

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
#
# Strategy: skip the 3 header rows, read everything, split into profile pairs.

with open(filename, 'r') as f:
    lines = f.readlines()

# Count how many profiles from line 0 (header row)
header = lines[0].split()
profile_names = [w for w in header if w.startswith('Profile')]
n_profiles = len(profile_names)
print(f"Found {n_profiles} profile(s): {', '.join(profile_names)}")

# Skip 3 header rows, read numeric data
profiles = {i: {'x': [], 'y': []} for i in range(n_profiles)}

for line in lines[3:]:
    parts = line.split()
    # Each profile occupies 2 columns; dashes mark missing data
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

# ── Plot ──────────────────────────────────────────────────────────────────────
colors = ['#185FA5', '#0F6E56', '#BA7517', '#A32D2D']

fig, ax = plt.subplots(figsize=(9, 4), facecolor='#f8f8f6')
ax.set_facecolor('#f8f8f6')

for p in range(n_profiles):
    x = profiles[p]['x']
    y = profiles[p]['y']
    if len(x) == 0:
        continue
    # zero-mean each profile for easier comparison
    y_zeroed = y - np.mean(y)
    ax.plot(x, y_zeroed, color=colors[p % len(colors)],
            lw=1.4, label=profile_names[p])

ax.axhline(0, color='#aaa', lw=0.6, ls='--')
ax.set_xlabel('Position (nm)', fontsize=12)
ax.set_ylabel('Height (nm)', fontsize=12)
ax.legend(fontsize=10, framealpha=0.6)
ax.grid(True, lw=0.4, alpha=0.4)
ax.tick_params(labelsize=10)

# ── Print roughness (Rq) per profile ─────────────────────────────────────────
print("\nRoughness summary:")
for p in range(n_profiles):
    y = profiles[p]['y']          # already in nm
    if len(y) == 0:
        continue
    rq   = np.std(y)
    ra   = np.mean(np.abs(y - np.mean(y)))
    rmax = np.max(y) - np.min(y)
    print(f"  {profile_names[p]}:  Rq = {rq:.3f} nm   Ra = {ra:.3f} nm   Rmax = {rmax:.3f} nm")

plt.tight_layout()

# Save next to the script, not back into the source folder
outname = os.path.basename(os.path.splitext(filename)[0]) + '_profile.png'
plt.savefig(outname, dpi=150, bbox_inches='tight')
print(f"\nSaved: {outname}")
plt.show()
