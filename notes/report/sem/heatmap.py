import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector



csvname = input("input your .csv file name here: ")
def heatmap(csvname):
    data = np.loadtxt(csvname, delimiter=",")

    #plot heatmap
    plt.figure(figsize=(8, 16))  # tall to match 1k x 2k aspect ratio
    plt.imshow(data, cmap="hot", aspect="auto")
    plt.colorbar(label="Electron Intensity")
    plt.title("Electron Intensity Heatmap")
    plt.xlabel("X (px)")
    plt.ylabel("Y (px)")
    plt.tight_layout()
    plt.savefig("heatmap.png", dpi=300)
    # plt.show()


    heatmap(csvname)