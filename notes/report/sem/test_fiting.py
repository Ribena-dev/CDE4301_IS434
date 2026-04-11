import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy.optimize import curve_fit
from scipy.special import erf
import os

# ── Load xlsx grid ─────────────────────────────────────────────
def load_xlsx_grid(filepath):
    df = pd.read_excel(filepath, sheet_name=0, header=None)
    # df = pd.read_csv(filepath, header=None)
    intensity = df.to_numpy().astype(float)
    print(f"Loaded grid: {intensity.shape}, min={intensity.min():.0f}, max={intensity.max():.0f}")
    return intensity

# ── Edge model ─────────────────────────────────────────────────
def edge_model(x, A, B, C, d, f):
    return (
        A * (1 + erf((2 * np.sqrt(np.log(2)) / f) * (d - x)))
        + B * np.exp(-(np.log(16) / f**2) * (d - x)**2)
        + C
    )

# ── Settings ───────────────────────────────────────────────────
filepath    = "1646 grid 256.xlsx"
h_nm        = 2000
nm_per_px   = 10               # update with your calibration
h_px        = h_nm / nm_per_px
results     = []
selection_count = [0]
os.makedirs("figures", exist_ok=True)
filename    = os.path.splitext(os.path.basename(filepath))[0]

intensity   = load_xlsx_grid(filepath)

# ── Fit profile ────────────────────────────────────────────────
def fit_profile(region, axis="x"):
    profile = region.mean(axis=0) if axis == "x" else region.mean(axis=1)
    pixels  = np.arange(len(profile))

    d_guess   = np.argmax(np.abs(np.diff(profile)))
    C_guess   = profile[d_guess:].mean()
    left_mean = profile[:d_guess].mean()
    A_guess   = (left_mean - C_guess) / 2
    B_guess   = profile.max() - left_mean

    p0     = [A_guess, B_guess, C_guess, d_guess, 10.0]
    bounds = (
        [0,    0,    0,   0,            1  ],
        [1000, 1000, 500, len(profile), 100]
    )

    try:
        popt, _ = curve_fit(edge_model, pixels, profile, p0=p0, bounds=bounds, maxfev=10000)
        A, B, C, d, f = popt
        theta = np.degrees(np.arctan(h_px / abs(f)))

        results.append({"axis": axis, "A": A, "B": B, "C": C, "d": d, "f": abs(f), "theta": theta})

        print(f"  A (erf amplitude):    {A:.2f}")
        print(f"  B (Gaussian amp):     {B:.2f}")
        print(f"  C (baseline):         {C:.2f}")
        print(f"  d (edge position px): {d:.2f}")
        print(f"  f (FWHM):             {abs(f):.2f} px")
        print(f"  θ (sidewall angle):   {theta:.2f}°")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(pixels, profile, "m.", label="Data")
        ax.plot(pixels, edge_model(pixels, *popt), "k-", label="Fit")
        ax.set_xlabel("Pixels")
        ax.set_ylabel("Counts")
        ax.set_title(f"Edge fit ({axis.upper()} scan) — FWHM: {abs(f):.2f} px | θ: {theta:.2f}°")
        ax.legend()
        plt.tight_layout()

        save_path = f"figures/{filename}_selection_{selection_count[0]}_{axis}.png"
        fig.savefig(save_path, dpi=300)
        print(f"Saved → {save_path}")
        plt.show()

    except RuntimeError:
        print("Fit failed — try selecting a region with a clearer edge")

# ── Save results ───────────────────────────────────────────────
def save_results():
    if not results:
        print("No results to save")
        return
    with open(f"fit_results_{filename}.txt", "w") as f_out:
        f_out.write("=== Individual Fits ===\n")
        for i, r in enumerate(results):
            f_out.write(f"\nSelection {i+1} ({r['axis'].upper()} scan):\n")
            f_out.write(f"  A:     {r['A']:.2f}\n")
            f_out.write(f"  B:     {r['B']:.2f}\n")
            f_out.write(f"  C:     {r['C']:.2f}\n")
            f_out.write(f"  d:     {r['d']:.2f} px\n")
            f_out.write(f"  f:     {r['f']:.2f} px\n")
            f_out.write(f"  theta: {r['theta']:.2f}°\n")
        f_out.write("\n=== Averages ===\n")
        for key in ["A", "B", "C", "d", "f", "theta"]:
            vals = [r[key] for r in results]
            f_out.write(f"  {key}: {np.mean(vals):.2f} ± {np.std(vals):.2f}\n")
    print(f"Saved {len(results)} results to fit_results_{filename}.txt")

# ── Selection callback ─────────────────────────────────────────
def on_select(eclick, erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    region  = intensity[y1:y2, x1:x2]
    selection_count[0] += 1
    print(f"\nSelection #{selection_count[0]}: x={x1}-{x2}, y={y1}-{y2}")
    fit_profile(region, axis="x")

# ── Main ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(intensity, cmap="gray", aspect="auto")
ax.set_title(f"{filename} — draw selection to fit edge")

selector = RectangleSelector(
    ax, on_select,
    useblit=True, button=[1],
    minspanx=5, minspany=5,
    spancoords="pixels", interactive=True
)

plt.show()
save_results()