import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import erf
from scipy.optimize import curve_fit

# 1. LOAD DATA
df = pd.read_excel('heatmaps/1646 grid 256.xlsx', header=None)
data_matrix = df.to_numpy()

# 2. SELECT ROW AND CROP FOR ONE EDGE
# row_idx = int(input("Enter row index to analyze: "))
full_row = data_matrix[:]

# Show the full row first to help you decide where to crop
plt.plot(full_row)
plt.title("Full Row Profile - Identify your crop points")
plt.show()

start_x = int(input("Enter start pixel for the single edge: "))
end_x = int(input("Enter end pixel for the single edge: "))

# Crop the data to focus on only ONE edge
y_data = full_row[start_x:end_x]
x_data = np.arange(len(y_data))

# 3. ERF-GAUSSIAN FUNCTION
# f = blur, h = step height, x0 = center, b = base offset
def erf_gaussian(x, f, h, x0, b):
    # Use + or - for rising vs falling edge. 
    # This version works for Rising edges. 
    # For Falling edges, change (1 + erf...) to (1 - erf...)
    return (h / 2) * (1 + erf((x - x0) / (f * np.sqrt(2)))) + b

# 4. PERFORM FIT
try:
    # Initial Guesses: [f=2, height, center_of_crop, min_val]
    p0 = [2.0, np.max(y_data)-np.min(y_data), len(y_data)/2, np.min(y_data)]
    
    popt, _ = curve_fit(erf_gaussian, x_data, y_data, p0=p0)
    f_measured = popt[0]
    
    # 5. CALC SIDEWALL ANGLE (Assuming 500nm Pd thickness)
    thickness = 500 
    theta = np.degrees(np.arctan(thickness / f_measured))

    # 6. PLOT
    plt.scatter(x_data, y_data, label='Edge Data', s=15, color='black', alpha=0.5)
    plt.plot(x_data, erf_gaussian(x_data, *popt), 'r--', label=f'Fit (f={f_measured:.2f}nm)')
    plt.title(f"Single Edge Analysis: {theta:.2f}° Sidewall")
    plt.xlabel("Relative Distance (nm)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()

    print(f"--- Results ---")
    print(f"Edge Blur (f): {f_measured:.3f} nm")
    print(f"Sidewall Angle: {theta:.2f} degrees")

except Exception as e:
    print(f"Error during fitting: {e}. Try adjusting your crop range.")