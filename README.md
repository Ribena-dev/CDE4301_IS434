# sample analysis tool guide

## Overview
creates a heatmap based on electron counts to prform sample analysis and data comparison.

## New Features

### 1.  Edge Function Fitting ✓

```
F = A[1 + Erf(2√(ln2)/f × (d-x))] + B×exp[-ln16/f² × (d-x)²] + C
```

**Parameters:**
- `A`: Amplitude of error function component (gradual signal increase at edge)
- `B`: Amplitude of Gaussian peak (local signal enhancement from SE emission)
- `C`: Baseline signal value (system background)
- `d`: Physical position of sharp edge on calibration grid
- `f`: Full Width at Half Maximum (FWHM) of beam spot
- `x`: Scanning position

**Features:**
- Automatic initial parameter estimation
- Bounded optimization for physical validity
- Residual analysis with R² calculation
- Parameter display on plot

---

### 2. Multi-Selection Comparison ✓
**Select multiple regions and compare their statistics side-by-side**

**How it works:**
1. Select Mode 2 from the menu
2. Click and drag to select first region (Selection 1)
3. Continue selecting additional regions (Selection 2, 3, 4...)
4. Close the plot window when done
5. View comprehensive comparison

**Comparison includes:**
- **Bar Chart**: Average counts for each selection
- **Statistics Table**: Mean, Std Dev, Min, Max, Size for each selection
- **Distribution Overlay**: Histogram comparison of value distributions
- **Summary Report**: Detailed statistics printed to console

**Use cases:**
- Compare signal intensity across different sample areas
- Identify regional variations in your data
- Quality control and consistency checking

---

### 3. Row-wise Set Analysis ✓
**Divide selected region into row sets and compare their line fits**

**How it works:**
1. Select Mode 3 from the menu
2. Click and drag to select a region (e.g., 100 rows × any columns)
3. Enter rows per set (e.g., 5 rows)
4. System creates sets (e.g., 100÷5 = 20 sets)
5. Each set is:
   - Averaged across its rows
   - Fitted with DLC edge function
   - Plotted for visual comparison

**Output includes:**
- **Individual Plots**: Each set shown separately with fit
- **Overlay Plot**: All sets and fits superimposed for direct comparison
- **Parameter Comparison**: Bar chart showing how A, B, C, d, f vary across sets

**Use cases:**
- Analyze signal variation along scanning direction
- Track edge position drift
- Identify systematic changes in beam parameters
- Compare different regions of your sample

---

## Usage Guide

### Running the Script

```bash
python heatmap_selector_enhanced.py
```

### Workflow

1. **Enter file path** when prompted
   ```
   Enter filepath and name: /path/to/your/data.xlsx
   ```

2. **Choose colormap** (or press Enter for default 'viridis')
   ```
   Choose a colour scheme: hot
   ```

3. **Select mode**:
   ```
   1. Single Selection (DLC Edge Fit)
   2. Multi-Selection Comparison
   3. Row-wise Set Analysis
   4. Exit
   ```

4. **Make your selection(s)** on the heatmap
   - Click and drag to draw rectangle
   - Release to process selection

5. **View results** in generated plots

6. **Continue or change modes** - the menu reappears after each analysis

---

## Mode Details

### Mode 1: Single Selection (DLC Edge Fit)
**Best for:** Detailed analysis of a single edge region

**Process:**
- Select region → Averages columns → Fits DLC function
- Shows fit quality, parameters, and residuals

**Output:**
- Top plot: Data points + DLC fit curve + parameters
- Bottom plot: Fit residuals + R² value

---

### Mode 2: Multi-Selection Comparison
**Best for:** Comparing multiple regions

**Process:**
- Select region 1 → adds to list
- Select region 2 → adds to list
- Continue as needed
- Close plot → view comparison

**Output:**
- Average counts bar chart
- Statistics table
- Distribution histograms
- Console summary

**Tips:**
- You can select as many regions as needed
- Selections are named automatically (Selection 1, 2, 3...)
- All statistics are calculated automatically

---

### Mode 3: Row-wise Set Analysis
**Best for:** Analyzing variation along one dimension

**Example workflow:**
```
Selected area: 100 rows × 50 columns
Enter number of rows per set: 5
→ Creates 20 sets (Set 1-20)
→ Each set averages 5 rows
→ Each averaged line is fitted with DLC function
→ All 20 fits displayed for comparison
```

**Output:**
- Grid of individual set plots (each with fit)
- Overlay comparison (all sets superimposed)
- Parameter comparison bar chart
- Console summary with all fit parameters

**Tips:**
- Choose rows per set based on your data density
- Smaller sets = more granular analysis
- Larger sets = smoother fits but less detail

---

## Technical Notes

###  Fitting Algorithm
- Uses `scipy.optimize.curve_fit` with bounded optimization
- Initial guesses based on data characteristics:
  - Baseline (C): minimum value
  - Error function amplitude (A): ~50% of range
  - Gaussian amplitude (B): ~25% of range
  - Edge position (d): location of maximum gradient
  - FWHM (f): ~10% of x-range

### Parameter Bounds
- All parameters ≥ 0 (physical constraint)
- Edge position (d) within data range
- FWHM (f) between 0.1 and full x-range

### Error Handling
- Minimum 5 points required for DLC fitting
- Falls back to data display if fitting fails
- Invalid selections are caught and reported

---

## Dependencies
```python
pandas
matplotlib
seaborn
numpy
openpyxl
scipy
```

Install if needed:
```bash
pip install pandas matplotlib seaborn numpy openpyxl scipy
```

---

## Example Use Cases

### Case 1: Edge Sharpness Analysis
**Goal:** Measure edge sharpness at different sample positions

**Method:**
1. Use Mode 3 (Row-wise Set Analysis)
2. Select region spanning edge
3. Set rows per set = 3-5 for high resolution
4. Compare 'f' (FWHM) values across sets
5. Lower f = sharper edge

---

### Case 2: Signal Uniformity Check
**Goal:** Verify uniform signal across sample

**Method:**
1. Use Mode 2 (Multi-Selection)
2. Select 5-10 regions across sample
3. Compare average counts
4. Check standard deviations
5. Identify outlier regions

---

### Case 3: Detailed Edge Characterization
**Goal:** Get precise edge parameters

**Method:**
1. Use Mode 1 (Single Selection)
2. Select narrow region around edge
3. Analyze all 5 DLC parameters
4. Check R² for fit quality
5. Use parameters for calibration

---

## Comparison: Original vs Enhanced

| Feature | Original | Enhanced |
|---------|----------|----------|
| Fitting Function | Polynomial (degree 5) | DLC edge function |
| Selection Mode | Single only | Three modes |
| Multiple Selections | No | Yes (Mode 2) |
| Set Analysis | No | Yes (Mode 3) |
| Residual Analysis | No | Yes |
| Parameter Display | Equation only | Full parameters + R² |
| Statistical Comparison | No | Yes (Mode 2) |
| Visual Comparison | Single plot | Multiple views |

---

## Tips & Best Practices

1. **For edge analysis:**
   - Select narrow region around edge (±10-20 pixels)
   - Ensure edge is roughly vertical in selection

2. **For set analysis:**
   - Rows per set should be 3-10 for stable fits
   - Too small = noisy fits, too large = loss of detail

3. **For multi-selection:**
   - Select regions of similar size for fair comparison
   - Avoid overlapping selections
   - Use consistent selection strategy

4. **General:**
   - Always check R² value (should be > 0.95 for good fits)
   - Review residual plots for systematic errors
   - Close plot windows to return to menu

---

## Troubleshooting

**"DLC fitting failed"**
- Not enough data points (need ≥5)
- Data too noisy → try larger selection
- Check data quality in original heatmap

**"Not enough rows"**
- Selected area smaller than rows per set
- Reduce rows per set or select larger area

**Fits look poor**
- Data may not follow DLC model
- Try different selection region
- Check if edge is actually present

**Plot windows not appearing**
- Backend issue → try: `export MPLBACKEND=TkAgg`
- Or add to script: `matplotlib.use('TkAgg')`

---

## Future Enhancements (Ideas)

- Export fit parameters to CSV
- Batch processing of multiple files
- Custom parameter bounds
- Alternative fitting functions
- 2D edge detection and fitting
- Automatic edge finding

---
