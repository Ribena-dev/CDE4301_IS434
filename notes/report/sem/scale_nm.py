import numpy as np
import tifffile as tiff
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy.optimize import curve_fit
from scipy.special import erf
import os

filename = "Grid-9e"
intensity = tiff.imread(filename + ".tif")
image_path = filename + ".tif"  # ← fix 1: .tff → .tif

# fix 3: convert RGB to grayscale if needed
if intensity.ndim == 3:
    intensity = intensity.mean(axis=2).astype(np.uint8)

print(intensity.shape, intensity.dtype)

def measure_scale_bar(image_path):
    """Click both ends of the scale bar to calculate and save nm_per_px."""
    scale_clicks = []
    scale_bar_nm = int(input("Enter scale bar length in nm: "))  # ask before showing

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(intensity, cmap="gray")
    ax.set_title("Click LEFT then RIGHT end of the scale bar")

    def on_click(event):
        if event.button == 1 and len(scale_clicks) < 2:
            scale_clicks.append(event.xdata)
            ax.axvline(event.xdata, color="red", linewidth=1)
            fig.canvas.draw()
            print(f"Click {len(scale_clicks)}: x = {event.xdata:.1f} px")

        if len(scale_clicks) == 2:
            scale_bar_px = abs(scale_clicks[1] - scale_clicks[0])
            nm_per_px = scale_bar_nm / scale_bar_px
            fname = os.path.basename(image_path)

            existing = {}
            if os.path.exists("nm_per_px.txt"):
                with open("nm_per_px.txt", "r") as f:
                    for line in f:
                        if ":" in line:
                            k, v = line.strip().split(":", 1)
                            existing[k.strip()] = v.strip()

            existing[fname] = f"{nm_per_px:.4f}"

            with open("nm_per_px.txt", "w") as f:
                for k, v in existing.items():
                    f.write(f"{k}: {v}\n")

            print(f"Scale bar: {scale_bar_px:.1f} px")
            print(f"nm/px:     {nm_per_px:.4f}")
            print(f"Saved →    {fname}: {nm_per_px:.4f}")
            plt.close()

    fig.canvas.mpl_connect("button_press_event", on_click)  # ← fix 2: connect BEFORE show
    plt.show()

measure_scale_bar(image_path)