#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import os
import re

# Get directory from command line or use current directory
directory = sys.argv[1] if len(sys.argv) > 1 else '.'

def read_transmit_file(filename):
    """Read Y, Z positions from TRANSMIT file"""
    y_pos, z_pos = [], []
    
    with open(filename) as f:
        for line in f:
            if line.startswith('T'):
                parts = line.split()
                y_pos.append(float(parts[5]))
                z_pos.append(float(parts[6]))
    
    return np.array(y_pos), np.array(z_pos)

def calc_spread(y, z):
    """Calculate average radial spread from center"""
    radial = np.sqrt(y**2 + z**2)
    return np.mean(radial)

# Read all txt files in folder
files = sorted(glob.glob(f'{directory}/*.txt'))
depths = []
spreads = []

for f in files:
    # Extract depth from filename - try to find numbers followed by 'um'
    filename = os.path.basename(f)
    match = re.search(r'(\d+(?:\.\d+)?)um', filename)
    if not match:
        print(f"Skipping {filename}: no depth found")
        continue
    depth = float(match.group(1))
    
    y, z = read_transmit_file(f)
    spread = calc_spread(y, z)
    
    depths.append(depth)
    spreads.append(spread)
    print(f"{f}: spread = {spread:.2f} Å")

# Convert spread from Angstroms to micrometers
spreads_um = np.array(spreads) / 10000

# Sort by depth
sorted_indices = np.argsort(depths)
depths = np.array(depths)[sorted_indices]
spreads_um = spreads_um[sorted_indices]

# Plot
plt.figure(figsize=(8, 6))
plt.plot(depths, spreads_um, 'o-', linewidth=2, markersize=8)
plt.xlabel('Depth (μm)', fontsize=12)
plt.ylabel('Average Spread (μm)', fontsize=12)
plt.title('Beam Spread vs Depth', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('beam_spread.png', dpi=150)
print("\nPlot saved to beam_spread.png")