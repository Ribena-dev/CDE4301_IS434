"""
SEM edge analysis — Erf + Gaussian fit → sidewall angle
=========================================================
Workflow:
  1. Load an Excel file of SEM intensity data (pixels × columns)
  2. Display it as a heatmap
  3. Click-drag to select a row range for the linescan
  4. Collapse selected rows (mean) → 1D intensity profile
  5. Fit the Erf + Gaussian model to each edge
  6. Report FWHM (f) and calculate sidewall angle θ = 90° − arctan(f/h)

Usage:
    python sem_edge_analysis.py
    → prompted for Excel file and feature height h (nm)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import SpanSelector
from scipy.optimize import curve_fit
from scipy.special import erf

# ── Model ─────────────────────────────────────────────────────────────────────

def erf_gauss(x, A, B, C, d, f):
    """
    Combined Erf + Gaussian edge model (van Kan / NIST formulation).

    F(x) = A[1 + Erf(2√ln2/f · (d-x))] + B·exp(-ln16/f²·(d-x)²) + C

    Parameters
    ----------
    A  : error function amplitude
    B  : Gaussian amplitude (secondary electron peak)
    C  : baseline intensity
    d  : edge position (nm or px)
    f  : FWHM of edge transition
    """
    arg  = (2 * np.sqrt(np.log(2)) / f) * (d - x)
    erf_term  = A * (1 + erf(arg))
    gauss_term = B * np.exp(-np.log(16) / f**2 * (d - x)**2)
    return erf_term + gauss_term + C


def fit_edge(x, y, edge_guess, label='edge'):
    """Fit the Erf+Gaussian model to one edge transition."""
    # Initial guesses
    amp   = (np.max(y) - np.min(y)) / 2
    p0    = [amp, amp * 0.3, np.min(y), edge_guess, 10.0]
    bounds = ([-np.inf, -np.inf, -np.inf, edge_guess - 50, 0.5],
              [ np.inf,  np.inf,  np.inf, edge_guess + 50, 200.0])
    try:
        popt, pcov = curve_fit(erf_gauss, x, y, p0=p0, bounds=bounds,
                               maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except RuntimeError:
        print(f"  Warning: fit did not converge for {label}")
        return None, None


# ── Load Excel ────────────────────────────────────────────────────────────────

def load_excel(path):
    df = pd.read_excel(path, header=None)
    data = df.values.astype(float)
    print(f"  Loaded: {data.shape[0]} rows × {data.shape[1]} columns")
    return data


# ── Interactive selection ─────────────────────────────────────────────────────

selected_rows = [None, None]

def run_selection(data):
    """Show heatmap; user selects a row range on the y-axis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, aspect='auto', cmap='hot',
                   origin='upper',
                   vmin=np.percentile(data, 2),
                   vmax=np.percentile(data, 98))
    plt.colorbar(im, ax=ax, label='Intensity (a.u.)')
    ax.set_xlabel('Column (px)', fontsize=11)
    ax.set_ylabel('Row (px)', fontsize=11)
    ax.set_title('SEM intensity heatmap\n'
                 'Drag on the LEFT EDGE (y-axis) to select row range, '
                 'then close this window', fontsize=10)

    rect_patch = [None]

    def onselect(y_min, y_max):
        r0, r1 = int(round(y_min)), int(round(y_max))
        r0 = max(0, r0); r1 = min(data.shape[0]-1, r1)
        selected_rows[0] = r0
        selected_rows[1] = r1

        if rect_patch[0] is not None:
            rect_patch[0].remove()
        rect_patch[0] = patches.Rectangle(
            (-0.5, r0), data.shape[1], r1 - r0,
            linewidth=1.5, edgecolor='cyan', facecolor='cyan', alpha=0.18)
        ax.add_patch(rect_patch[0])
        ax.set_title(f'Selected rows {r0}–{r1}  '
                     f'({r1-r0+1} rows)  — close window when happy', fontsize=10)
        fig.canvas.draw_idle()

    span = SpanSelector(ax, onselect, 'vertical',
                        useblit=True,
                        props=dict(facecolor='cyan', alpha=0.25))

    plt.tight_layout()
    plt.show()

    return selected_rows[0], selected_rows[1]


# ── Fit and report ────────────────────────────────────────────────────────────

def analyse(data, r0, r1, h_nm, nm_per_px):
    rows = data[r0:r1+1, :]
    n_raw = rows.shape[0]

    # Remove outlier rows using IQR x3 on each row's mean before collapsing
    row_means = rows.mean(axis=1)
    q1, q3 = np.percentile(row_means, 25), np.percentile(row_means, 75)
    iqr = q3 - q1
    mask = (row_means >= q1 - 3*iqr) & (row_means <= q3 + 3*iqr)
    n_removed = int(np.sum(~mask))
    if n_removed > 0:
        print(f"  Outlier rows removed: {n_removed} of {n_raw} "
              f"({100*n_removed/n_raw:.1f}%)")
    else:
        print(f"  No outlier rows detected ({n_raw} rows used)")
    profile = np.mean(rows[mask], axis=0)
    x = np.arange(len(profile), dtype=float)

    # Locate approximate edges from gradient
    grad = np.abs(np.gradient(profile))
    # Take the two highest-gradient peaks as left and right edge
    peak_idxs = []
    remaining = grad.copy()
    for _ in range(2):
        idx = int(np.argmax(remaining))
        peak_idxs.append(idx)
        lo, hi = max(0, idx-15), min(len(remaining), idx+15)
        remaining[lo:hi] = 0
    peak_idxs.sort()

    if len(peak_idxs) < 2:
        print("Could not detect two edges automatically.")
        return

    left_edge, right_edge = peak_idxs

    print(f"\nDetected edges at columns ~{left_edge} and ~{right_edge} px")
    print(f"Approximate CD = {right_edge - left_edge} px  "
          f"= {(right_edge - left_edge)*nm_per_px:.1f} nm")

    # Fit each edge in a local window
    win = 60
    results = {}
    for side, guess in [('left', left_edge), ('right', right_edge)]:
        lo = max(0, guess - win)
        hi = min(len(x), guess + win)
        xw, yw = x[lo:hi], profile[lo:hi]
        popt, perr = fit_edge(xw, yw, float(guess), label=side)
        if popt is not None:
            A, B, C, d, f = popt
            results[side] = dict(A=A, B=B, C=C, d=d, f=f,
                                 popt=popt, xw=xw, yw=yw)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor='#f8f8f6')
    for ax in axes: ax.set_facecolor('#f8f8f6')

    # Full collapsed profile
    axes[0].plot(x, profile, color='#185FA5', lw=1.4)
    for side, col in [('left','#A32D2D'), ('right','#0F6E56')]:
        if side in results:
            axes[0].axvline(results[side]['d'], color=col, lw=1.5,
                            ls='--', label=f'{side} edge')
    axes[0].set_xlabel('Column (px)', fontsize=11)
    axes[0].set_ylabel('Intensity (a.u.)', fontsize=11)
    axes[0].set_title(f'Collapsed linescan (rows {r0}–{r1})', fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, lw=0.3, alpha=0.4)

    # Left edge fit
    if 'left' in results:
        r = results['left']
        x_fit = np.linspace(r['xw'][0], r['xw'][-1], 400)
        axes[1].scatter(r['xw'], r['yw'], s=12, color='#185FA5', alpha=0.7,
                        label='data')
        axes[1].plot(x_fit, erf_gauss(x_fit, *r['popt']),
                     color='#A32D2D', lw=2, label='fit')
        axes[1].axvline(r['d'], color='#A32D2D', lw=1, ls='--')
        axes[1].set_title(f"Left edge  f = {r['f']:.1f} px "
                          f"= {r['f']*nm_per_px:.1f} nm", fontsize=10)
        axes[1].set_xlabel('Column (px)', fontsize=10)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, lw=0.3, alpha=0.4)

    # Right edge fit
    if 'right' in results:
        r = results['right']
        x_fit = np.linspace(r['xw'][0], r['xw'][-1], 400)
        axes[2].scatter(r['xw'], r['yw'], s=12, color='#185FA5', alpha=0.7,
                        label='data')
        axes[2].plot(x_fit, erf_gauss(x_fit, *r['popt']),
                     color='#0F6E56', lw=2, label='fit')
        axes[2].axvline(r['d'], color='#0F6E56', lw=1, ls='--')
        axes[2].set_title(f"Right edge  f = {r['f']:.1f} px "
                          f"= {r['f']*nm_per_px:.1f} nm", fontsize=10)
        axes[2].set_xlabel('Column (px)', fontsize=10)
        axes[2].legend(fontsize=8)
        axes[2].grid(True, lw=0.3, alpha=0.4)

    plt.tight_layout()
    plt.savefig('sem_edge_fit.png', dpi=150, bbox_inches='tight')
    print("\nSaved: sem_edge_fit.png")
    plt.show()

    print("\n" + "─"*50)
    print("Results")
    print("─"*50)
    print(f"Feature height h    = {h_nm} nm")
    print(f"Pixel size          = {nm_per_px} nm/px")
    print()

    f_vals_nm = []
    for side in ['left', 'right']:
        if side not in results:
            continue
        r = results[side]
        f_px = r['f']
        f_nm = f_px * nm_per_px
        f_vals_nm.append(f_nm)
        theta = 90 - np.degrees(np.arctan(f_nm / h_nm))
        print(f"  {side.capitalize()} edge:")
        print(f"    Edge position d  = {r['d']:.1f} px  = {r['d']*nm_per_px:.1f} nm")
        print(f"    FWHM f           = {f_px:.2f} px  = {f_nm:.2f} nm")
        print(f"    A (Erf amp)      = {r['A']:.3f}")
        print(f"    B (Gauss amp)    = {r['B']:.3f}")
        print(f"    C (baseline)     = {r['C']:.3f}")
        print(f"    Sidewall angle θ = 90° − arctan({f_nm:.2f}/{h_nm}) "
              f"= {theta:.2f}°")
        print()

    if len(f_vals_nm) == 2:
        f_mean_nm = np.mean(f_vals_nm)
        theta_mean = 90 - np.degrees(np.arctan(f_mean_nm / h_nm))
        print(f"  Mean FWHM        = {f_mean_nm:.2f} nm")
        print(f"  Mean θ           = {theta_mean:.2f}°")
    print("─"*50)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    path = input("Excel file path: ").strip()
    if not os.path.exists(path):
        print(f"File not found: {path}"); sys.exit(1)

    h_nm      = float(input("Feature height h (nm): ").strip())
    nm_per_px = float(input("Pixel size (nm/px): ").strip())

    print(f"\nLoading {os.path.basename(path)} ...")
    data = load_excel(path)

    print("\nStep 1 — Select row range on the heatmap (drag on y-axis).")
    r0, r1 = run_selection(data)

    if r0 is None:
        print("No rows selected — using all rows.")
        r0, r1 = 0, data.shape[0] - 1

    print(f"\nStep 2 — Fitting edges on rows {r0}–{r1} ...")
    analyse(data, r0, r1, h_nm, nm_per_px)


if __name__ == '__main__':
    main()