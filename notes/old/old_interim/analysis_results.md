## 5. Results and Analysis

### 5.1 Visible Structural Defects

Obvious curvature along the sidewalls became apparent during my inspection of the fabricated structures. While one possible explanation is that system noise may be introducing a consistent, systematic curvature, the complexity of the beamline and control electronics places such troubleshooting beyond the scope of what I can verify directly.

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/defect.png" alt="Defect pattern" width="250">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 5.1:</strong> visible structural defect seen with optical microscope
  </figcaption>
</figure>


Another explanation I am exploring concerns the beam-writing strategy itself. For this sample, I used a high dosage with a single pass, assuming that the step size was sufficiently small for dose averaging to remain unaffected. However, the repeated curvature pattern suggests that this assumption may not hold. To test this, I plan to fabricate a second set of samples using a lower dosage applied over multiple passes. This approach may better average out dose variations, although it also carries the risk of introducing drift-related artefacts between passes.

### 5.2 Edge Analysis and FWHM Methodology

To validate my analysis pipeline, I first applied the automated Python implementation of the FWHM-based edge extraction method to a previously fabricated nickel grid. The current gold–DLC sample will undergo the same analysis once imaging is complete. This method allows me to quantify edge straightness and estimate the beam’s effective spot size directly from SEM intensity profiles. The fitting model combines an error function to capture the edge transition with a Gaussian term representing beam-induced spreading [17]:

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/formula.png" alt="formula" width="600">
</figure>


<figure style="text-align: center; margin: 20px 0;">
  <img src="images/nm_grid.png" alt="nickle grid" width="250">
   <img src="line_fited_20.png" alt="nickle grid graph" width="250">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 5.2 and 5.3:</strong> previously made nickel grid analysis
  </figcaption>
</figure>

From analysis of representative edge (20 collapsed rows), edge position is 1675.7 nm from image origin, beam spot FWHM is 11.7 nm, error function amplitude is , Gaussian amplitude is 1.5 indicating minimal beam tail effects, and baseline is 0.66.

A python script was developed for more automated analysis, the software overview can be found in annex A.
### 5.3 DLC and Gold Coating Contrast

DLC and gold produces contrast ratio of roughly **0.6** at 10 nm DLC height; lower ratio is better. After 10 nm, no substantial increase or decrease in contrast ratio occurs. Currently, insufficient sampling prevents quantitatively validating this claim, but if DLC to gold contrast stays at 0.6, testing different metal composition may be necessary.


[← Fabrication steps](fabrication.md) | [Next: Next steps and plans →](next_steps.md)
