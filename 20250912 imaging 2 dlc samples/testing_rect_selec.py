import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector


np.random.seed(42)
x_data = np.random.rand(100)
y_data = np.random.rand(100)

def onselect(eclick,erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print(f"Selected rectangle coordinates: ({x1}, {y1}) to ({x2}, {y2})")

fig, ax = plt.subplots()
ax.scatter(x_data, y_data)
# Define the RectangleSelector
rect_selector = RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels')
plt.show()