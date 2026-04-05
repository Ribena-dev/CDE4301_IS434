"""
Heatmap selector with edge fitting
====================================
1. Load Excel file
2. Display heatmap
3. Draw a rectangle to select a region
4. Collapse selected rows to mean profile
5. Fit Erf + Gaussian to edge
6. Report f, theta

Usage:  python heatmap_selector.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector
from scipy.optimize import curve_fit
from scipy.special import erf
from openpyxl import load_workbook

BG = '#f8f8f6'
B, T, R, A = '#185FA5', '#0F6E56', '#A32D2D', '#BA7517'

# ── Load ──────────────────────────────────────────────────────────────────────

def load(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    data = np.array(
        [[v if isinstance(v, (int, float)) else 0 for v in row]
         for row in ws.iter_rows(values_only=True)], dtype=float)
    wb.close()
    print(f"  Loaded {data.shape[0]} rows x {data.shape[1]} cols"
          f"  min={data.min():.0f}  max={data.max():.0f}")
    return data

# ── Erf + Gaussian model ──────────────────────────────────────────────────────

def erf_gauss(x, A_amp, B_amp, C, d, f):
    z = (2 * np.sqrt(np.log(2)) / f) * (d - x)
    return A_amp * (1 + erf(z)) + B_amp * np.exp(-np.log(16)/f**2 * (d-x)**2) + C

# ── Fit ───────────────────────────────────────────────────────────────────────

def fit_edge(x_nm, y):
    d_guess = float(x_nm[np.argmax(np.abs(np.gradient(y)))])
    amp     = (y.max() - y.min()) / 2
    p0      = [amp, amp * 0.3, y.min(), d_guess, 8.0]
    bounds  = ([-np.inf, 0, -np.inf, x_nm.min(), 0.5],
               [ np.inf, np.inf, np.inf, x_nm.max(), (x_nm.max()-x_nm.min())/2])
    try:
        popt, _ = curve_fit(erf_gauss, x_nm, y, p0=p0,
                            bounds=bounds, maxfev=10000)
        return popt
    except Exception as e:
        print(f"  Fit failed: {e}")
        return None

# ── Main ──────────────────────────────────────────────────────────────────────

path       = input("Excel file path: ").strip()
nm_per_px  = float(input("Pixel size (nm/px): ").strip())
h_nm       = float(input("Feature height h (nm): ").strip())
cmap       = input("Colourmap (hot / viridis / cividis) [hot]: ").strip() or 'hot'

if not os.path.exists(path):
    print("File not found."); sys.exit(1)

data = load(path)

# ── Heatmap + rectangle selector ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG)
ax.set_facecolor(BG)
im = ax.imshow(data, cmap=cmap, aspect='auto', origin='upper',
               vmin=np.percentile(data, 2), vmax=np.percentile(data, 98))
plt.colorbar(im, ax=ax, label='Count', shrink=0.85)
ax.set_xlabel('Column (px)', fontsize=10)
ax.set_ylabel('Row (px)', fontsize=10)
ax.set_title('Draw a rectangle across the edge — close window when done', fontsize=10)

sel = {}

def onselect(eclick, erelease):
    x1, x2 = sorted([eclick.xdata, erelease.xdata])
    y1, y2 = sorted([eclick.ydata, erelease.ydata])
    sel['col'] = (max(0, int(x1)), min(data.shape[1]-1, int(x2)))
    sel['row'] = (max(0, int(y1)), min(data.shape[0]-1, int(y2)))
    ax.set_title(f"Rows {sel['row'][0]}-{sel['row'][1]}  "
                 f"Cols {sel['col'][0]}-{sel['col'][1]}  — close when happy",
                 fontsize=10)
    fig.canvas.draw_idle()

rs = RectangleSelector(ax, onselect, useblit=False, button=[1],
                       interactive=True, drag_from_anywhere=True,
                       props=dict(facecolor='cyan', edgecolor='cyan',
                                  alpha=0.25, linewidth=1.5))
plt.tight_layout()
plt.show()

if not sel:
    print("No selection made."); sys.exit(0)

r0, r1 = sel['row']
c0, c1 = sel['col']

# ── Collapse + fit ────────────────────────────────────────────────────────────

region  = data[r0:r1+1, c0:c1+1]
profile = np.mean(region, axis=0)
x_px    = np.arange(c0, c1+1, dtype=float)
x_nm    = x_px * nm_per_px

popt = fit_edge(x_nm, profile)

# ── Results plot ──────────────────────────────────────────────────────────────

fig2, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=BG)
for ax in axes:
    ax.set_facecolor(BG)

# Left: full profile
axes[0].plot(x_nm, profile, color=B, lw=1.4)
axes[0].set_xlabel('Position (nm)', fontsize=10)
axes[0].set_ylabel('Mean intensity', fontsize=10)
axes[0].set_title(f'Collapsed profile  rows {r0}-{r1}', fontsize=10)
axes[0].grid(True, lw=0.3, alpha=0.4)

# Right: edge fit
axes[1].scatter(x_nm, profile, s=12, color=B, alpha=0.6, label='data')

if popt is not None:
    A_amp, B_amp, C, d_nm, f_nm = popt
    x_fit = np.linspace(x_nm[0], x_nm[-1], 500)
    axes[1].plot(x_fit, erf_gauss(x_fit, *popt), color=R, lw=2, label='Erf+Gauss fit')
    axes[1].axvline(d_nm, color=R, lw=1, ls='--', alpha=0.7)
    theta = 90 - np.degrees(np.arctan(f_nm / h_nm))
    axes[1].set_title(f'f = {f_nm:.2f} nm   θ = {theta:.2f}°', fontsize=10)

    axes[0].axvline(d_nm, color=R, lw=1, ls='--', alpha=0.6, label='edge')
    axes[0].legend(fontsize=8)

    print("\n" + "─"*44)
    print("Results")
    print("─"*44)
    print(f"  Pixel size       = {nm_per_px} nm/px")
    print(f"  Feature height h = {h_nm} nm")
    print(f"  Edge position d  = {d_nm:.1f} nm")
    print(f"  FWHM f           = {f_nm:.2f} nm")
    print(f"  A (Erf amp)      = {A_amp:.3f}")
    print(f"  B (Gauss amp)    = {B_amp:.3f}")
    print(f"  C (baseline)     = {C:.3f}")
    print(f"  Sidewall angle θ = {theta:.2f}°")
    print(f"  Meets >=89.4°    = {'YES' if theta >= 89.4 else 'NO'}")
    print("─"*44)
else:
    axes[1].set_title('Fit failed — check selection spans the edge', fontsize=10)

axes[1].set_xlabel('Position (nm)', fontsize=10)
axes[1].set_ylabel('Mean intensity', fontsize=10)
axes[1].legend(fontsize=8)
axes[1].grid(True, lw=0.3, alpha=0.4)

plt.tight_layout()
out = os.path.splitext(os.path.basename(path))[0] + '_fit.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.show()