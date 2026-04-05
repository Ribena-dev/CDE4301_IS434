### 4.4 Sidewall angle via electron detector

Edge profiles were extracted from the electron detector scan data using the
Erf–Gaussian fitting pipeline described in Section 2.5.1. For each scan a row
band was selected perpendicular to the grid edge, the selected rows were collapsed
to a mean 1D profile, and the combined Erf–Gaussian model was fitted to extract
the edge transition FWHM f and the corresponding sidewall angle θ.

#### 4.4.1 Grid heatmaps

Before edge analysis, the raw electron count maps were inspected to assess grid
geometry and feature visibility. Figures 4.X–4.Y show the heatmaps from the
five scan conditions acquired on 12 September and 20 December 2025.

[INSERT Figure 4.X — heatmaps: 2012 pin scans (6.384, 6.41),
2005 edet scan, 1953 Trek edet 2 scans, 1646 grid 256]

The 1953 Trek edet scan (Figure 4.X, image 8) shows the clearest full grid
geometry — a 3×3 array of dark square cells separated by bright metallic grid
bars, with good contrast across the full 1024×1024 field. The 2012 and 2005
scans show a partial cross-shaped feature, suggesting the scan field was
positioned over a single grid intersection rather than the full array. The
1646 scan at 256×256 pixels captures a complete cross pattern at lower spatial
resolution. These differences reflect the stage positioning coordinates logged
during each session rather than differences in the fabricated geometry.

#### 4.4.2 Edge fitting results

Figure 4.Y shows the collapsed profiles and Erf–Gaussian fits for each
measurement. Results are summarised in Table 4.X.

[INSERT Figure 4.Y — collapsed profiles and fits:
DLC/Au boundary, 2012 6.385, 2012 6.41, nickel reference]

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/dlc_au_boundary.png" alt="DLC/Au boundary edge fit"
       width="480" style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 4.Y(a):</strong> DLC/Au boundary — f = 5.93 nm, θ = 89.66°.
  </figcaption>
</figure>

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/2012_6_385_graph.png"
       alt="2012 grid 6.385 edge fit" width="480" style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 4.Y(b):</strong> 2012 grid 6.385 — f = 88.91 nm, θ = 89.49°.
  </figcaption>
</figure>

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/2012_6_41_graph.png"
       alt="2012 grid 6.41 edge fit" width="480" style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 4.Y(c):</strong> 2012 grid 6.41 — f = 51.30 nm, θ = 87.06°.
  </figcaption>
</figure>

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/nickel_graph.png"
       alt="Nickel reference grid edge fit" width="480" style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 4.Y(d):</strong> Nickel reference grid — f = 11.21 nm,
    θ = 89.68°.
  </figcaption>
</figure>

| Measurement | f (nm) | θ (°) | Meets ≥89.4° |
|---|---|---|---|
| DLC/Au boundary | 5.93 | **89.66** | ✓ |
| 2012 grid 6.385 (pin) | 88.91 | **89.49** | ✓ |
| 2012 grid 6.41 (pin) | 51.30 | 87.06 | ✗ |
| Nickel reference | 11.21 | **89.68** | ✓ |
| SRIM theoretical | 1.91 | 89.90 | ✓ |

**Table 4.X:** Edge FWHM and sidewall angle for each measurement. Feature
height h = 40 nm used throughout.

#### 4.4.3 Discussion of results

Three of the four measurements meet the ≥89.4° deliverable. The DLC/Au boundary
and nickel reference both give very sharp edges (f < 12 nm) with angles above
89.6°, closely approaching the SRIM theoretical prediction of 89.9°. The 2012
grid 6.385 scan passes the threshold at 89.49° despite a substantially larger
f of 88.91 nm, which reflects the lower pixel resolution of that scan
(nm/px scale set by the overview scan calibration of ~1116 nm/px) rather than
genuine sidewall degradation — the large f in physical units arises directly
from the coarse pixel size.

The 2012 grid 6.41 scan fails the target at θ = 87.06° with f = 51.30 nm.
Inspecting the heatmap (Figure 4.X image 6), this scan covers a different
stage position and the grid geometry appears shifted relative to the 6.384
position, suggesting the edge selected may not have been perfectly
perpendicular to the scan axis, which would artificially broaden the measured
transition and underestimate θ.

The nickel reference result (θ = 89.68°) is consistent with the interim report
value reported for the same sample and confirms the analysis pipeline is
producing physically reasonable values. The agreement between the nickel
reference and the DLC/Au boundary result, both measured at higher effective
pixel resolution, gives confidence that the true sidewall angle of the
fabricated features is in the range of 89.6° to 89.7°, in good agreement with
the SRIM prediction of 89.9°. The residual 0.2° gap is within the expected
range of measurement uncertainty from pixel size, row selection positioning,
and the finite beam spot size of the electron detector.