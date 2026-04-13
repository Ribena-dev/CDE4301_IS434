import numpy as np
import tifffile as tiff
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy.optimize import curve_fit
from scipy.special import erf
import os



filename = "Grid-12"
intensity = tiff.imread(filename+".tif")
print(intensity.shape,intensity.dtype)
csvname = "intensity_" + filename + ".csv"
np.savetxt(csvname, intensity,delimiter = ",", fmt = "%d")
selection_count = [0]
results=[]


h = 100 #nm



def load_nm_per_px(image_path):
    """Look up nm_per_px for the given image file from nm_per_px.txt."""
    filename = image_path

    if not os.path.exists("nm_per_px.txt"):
        print("nm_per_px.txt not found — run measure_scale_bar() first")
        return None

    with open("nm_per_px.txt", "r") as f:
        for line in f:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                if k.strip() == filename:
                    nm_per_px = float(v.strip())
                    print(f"Loaded nm/px for {filename}: {nm_per_px:.4f}")
                    return nm_per_px

    print(f"No entry found for {filename} in nm_per_px.txt")
    return None

nm_per_px = load_nm_per_px(filename+"e.tif")
print(nm_per_px)
h_px = h/nm_per_px

def edge_model(x, A, B, C, d, f):
    """Combined error function + Gaussian model"""
    return (
        A * (1 + erf((2 * np.sqrt(np.log(2)) / f) * (d - x)))
        + B * np.exp(-(np.log(16) / f**2) * (d - x)**2)
        + C
    )
def save_results():
    if not results:
        print("No results to save")
        return

    with open("fit_results_"+filename+".txt", "w") as f_out:
        f_out.write("=== Individual Fits ===\n")
        for i, r in enumerate(results):
            f_out.write(f"\nSelection {i+1} ({r['axis'].upper()} scan):\n")
            f_out.write(f"  A (erf amplitude):    {r['A']:.2f}\n")
            f_out.write(f"  B (Gaussian amp):     {r['B']:.2f}\n")
            f_out.write(f"  C (baseline offset):  {r['C']:.2f}\n")
            f_out.write(f"  d (edge position px): {r['d']:.2f}\n")
            f_out.write(f"  f (FWHM):             {r['f']:.2f} px\n")
            f_out.write(f"  theta :             {r['theta']:.2f}°\n")


        # Averages
        f_out.write("\n=== Averages across all selections ===\n")
        for key in ["A", "B", "C", "d", "f","theta"]:
            vals = [r[key] for r in results]
            f_out.write(f"  {key}: {np.mean(vals):.2f} ± {np.std(vals):.2f}\n")

        print(f"Saved {len(results)} results to fit_results_"+filename+".txt")


def fit_profile(region, axis="x"):
    # Collapse region to 1D line profile by averaging along the other axis
    if axis == "x":
        profile = region.mean(axis=0)  # average rows → profile along x
    else:
        profile = region.mean(axis=1)  # average cols → profile along y

    pixels = np.arange(len(profile))

    d_guess = np.argmax(np.abs(np.diff(profile)))  # edge = steepest gradient
    C_guess = profile[d_guess:].mean()             # baseline (right of edge)
    left_mean = profile[:d_guess].mean()           # plateau (left of edge)
    A_guess = (left_mean - C_guess) / 2            # step amplitude
    B_guess = profile.max() - left_mean            # Gaussian peak above plateau
    f_guess = 10.0                                 # wider FWHM starting point

    p0 = [A_guess, B_guess, C_guess, d_guess, f_guess]
    try:
        popt, _ = curve_fit(edge_model, pixels, profile, p0=p0, maxfev=10000)
        A, B, C, d, f = popt


        theta = None
        if h_px is not None:
            print(h_px)
            theta = 90 - np.degrees(np.arctan(abs(f) / h_px))
           

        results.append({"axis": axis, "A": A, "B": B, "C": C, "d": d, "f": abs(f), "theta": theta})

        print(f"  A (erf amplitude):    {A:.2f}")
        print(f"  B (Gaussian amp):     {B:.2f}")
        print(f"  C (baseline offset):  {C:.2f}")
        print(f"  d (edge position px): {d:.2f}")
        print(f"  f (FWHM):             {f:.2f} px")
        print(f"  θ (sidewall angle):   {theta:.2f}°")

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(pixels, profile, "m.", label="Data")
        ax.plot(pixels, edge_model(pixels, *popt), "k-", label="Fit")
        ax.set_xlabel("Pixels")
        ax.set_ylabel("Counts")
        ax.set_title(f"Edge fit ({axis.upper()} scan) — FWHM: {abs(f):.2f} px" + 
                     (f" | θ: {theta:.2f}°" if theta is not None else ""))
        ax.legend()
        plt.tight_layout()

        save_path = f"figures/{filename}_selection_{selection_count[0]}_{axis}.png"
        fig.savefig(save_path, dpi=300)
        print(f"Saved figure → {save_path}")


        plt.show()

    except RuntimeError:
        print("Fit failed — try selecting a region with a clearer edge")


def plot_tif(intensity):
    fig, ax = plt.subplots(figsize=(8, 16))
    ax.imshow(intensity, cmap="gray", aspect="auto")

    selector = RectangleSelector(
    ax, on_select,
    useblit=True,
    button=[1],          # left click only
    minspanx=5, minspany=5,
    spancoords="pixels",
    interactive=True)

    plt.show()
    save_results()
    # ax.set_title("Click and drag to select a region")


def on_select(eclick,erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    
    
    # Crop the selected region
    region = intensity[y1:y2, x1:x2]
    selection_count[0] += 1
    
    print(f"Selected region: x={x1}-{x2}, y={y1}-{y2}")
    print(f"Mean intensity: {region.mean():.2f}")
    print(f"Max intensity:  {region.max()}")
    print(f"Min intensity:  {region.min()}")

    print("\n--- X line scan ---")
    fit_profile(region, axis="x")
    print("\n--- Y line scan ---")
    fit_profile(region, axis="y")
    
    
    # Optionally save just the selected region
    selected_file = "selected_" +filename + ".csv"
    np.savetxt(selected_file, region, delimiter=",", fmt="%d")


plot_tif(intensity)
