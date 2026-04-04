## Results and analysis
### 4.1 General overview
<figure style="text-align: center; margin: 20px 0;">
  <img src="images/dosage.png" alt="resolution fabrication overview"  style="margin: 5px;">
  
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 3.1:</strong> Fabrication process for resolution standard overview
  </figcaption>
</figure>
 
Proton beam writing doses ranged from 75–175 nC/mm². All samples underwent electron detector edge analysis and AFM surface roughness measurement.
Samples [X] and [Y] showed the most promising initial optical inspection results
and were prioritised for detailed characterisation.

### 4.2 Comparing electron contrast

The backscatter contrast ratio between the metal grid features and the silicon substrate was assessed from the electron detector intensity profiles. A higher contrast ratio indicates greater separation between the metal signal and the background, which is the primary functional requirement of the resolution standard for SEM calibration use.

[INSERT electron detector heatmap side-by-side comparison across samples]

| Sample | Metal (top layer) | Z | Mean signal (a.u.) | Background (a.u.) | Contrast ratio |
|---|---|---|---|---|---|
| 1 | Au | 79 | [TBC] | [TBC] | [TBC] |
| 2 | DLC | 6 | [TBC] | [TBC] | [TBC] |
| 3 | DLC/Pd | 6/46 | [TBC] | [TBC] | [TBC] |
| 4 | Au/Pd | 79 | [TBC] | [TBC] | [TBC] |
| 5 | Pd | 46 | [TBC] | [TBC] | [TBC] |
| 6 | DLC/Pd/Ti | 6/46 | [TBC] | [TBC] | [TBC] |
| 7 | Pd/Ti | 46 | [TBC] | [TBC] | [TBC] |

The theoretical expectation is that Au produces the highest contrast against Si , followed by Pd  and DLC 


### 4.3 Surface roughness

Surface roughness Rq was measured by AFM in tapping mode. Two regions were characterised for each sample: the top face of the metal grid feature, and the exposed silicon substrate between features.

[INSERT AFM height map of best and worst performing samples side-by-side]

| Sample | Metal surface Rq (nm) | Si substrate Rq (nm) | Meets <3 nm target |
|---|---|---|---|
| 1 | [TBC] | [TBC] | [TBC] |
| 2 | [TBC] | [TBC] | [TBC] |
| 3 | [TBC] | [TBC] | [TBC] |
| 4 | [TBC] | [TBC] | [TBC] |
| 5 | [TBC] | [TBC] | [TBC] |
| 6 | [TBC] | [TBC] | [TBC] |
| 7 | [TBC] | [TBC] | [TBC] |

[INSERT Névot–Croce plot with measured Rq values marked]

Au deposited by magnetron sputtering was expected to show the highest roughness due to grain nucleation during island growth — consistent with the AFM observations from the interim report (Rq ≈ 3.3 nm). Pd deposited by e-beam evaporation was expected to produce smoother films due to the more directional, lower-energy deposition flux. 

[INSERT comment on measured values vs prediction.]


[INSERT comparison against measured values and comment on whether
results are consistent with Z-number prediction.]

### 4.4 Sidewall angle via electron detector

Edge profiles were extracted from the electron detector data for each sample using the Erf–Gaussian fitting pipeline described in Section 2.5.1. For each sample, a row band was selected over a single grid edge, individual row profiles were fitted independently, and the mean FWHM f and sidewall angle θ were reported with ±1σ across rows.

[INSERT edge_profiles.png — overlaid row profiles + mean for best sample]
[INSERT edge_sidewall.png — θ per row with mean ± 1σ and 89° target line]

| Sample | Pixel size X (nm/px) | f mean (nm) | f std (nm) | θ mean (°) | θ std (°) | Meets ≥89.4° |
|---|---|---|---|---|---|---|
| 1 | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |
| 2 | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |
| 3 | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |
| 4 | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |
| 5 | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |
| 6 | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |
| 7 | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |

SRIM theoretical prediction: θ = 89.9° (f = 1.91 nm at h = 1000 nm).

[INSERT comment on deviation from theoretical prediction and likely causes.]

### 4.5 Comparison across samples

A summary comparison of all five key metrics across all seven samples is presented below. Samples are ranked by sidewall angle as the primary deliverable, with surface roughness and contrast ratio as secondary criteria.

[INSERT summary radar chart or bar chart comparing θ and Rq across samples]

| Sample | θ mean (°) | Rq (nm) | Contrast ratio | Lift-off quality | Overall |
|---|---|---|---|---|---|
| 1 | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |
| 2 | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |
| 3 | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |
| 4 | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |
| 5 | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |
| 6 | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |
| 7 | [TBC] | [TBC] | [TBC] | [TBC] | [TBC] |

**Best performing sample: [TBC]**

[INSERT SEM/electron detector image + AFM map of best sample side-by-side]

This sample achieved θ = [TBC]° against the ≥89.4° target and Rq = [TBC] nm against the <3 nm target. [INSERT one sentence on why this composition outperformed the others — e.g. adhesion layer reducing grain stress, e-beam vs sputtering deposition mode, etc.]



### 4.6 Discussion

#### Deviation from SRIM prediction

The SRIM simulation predicted a theoretical sidewall angle of 89.9° based on a lateral straggle of σ = 0.81 nm at 1 µm depth. Measured values of θ = [TBC]° represent a deviation of [TBC]°. Likely sources of this deviation include: 

- **Development conditions**: overdevelopment or underdevelopment in the DI:IPA  developer can widen or narrow the trench beyond the exposed region, adding or removing material from the sidewall and shifting the apparent edge position. Additionally, PMMA has a melting point of bellow 100 degrees i is highly possible the side wall was damaged during metal sputtering 

#### Metal and adhesion layer effects

[INSERT comment on whether Pd/Ti samples outperformed Au/Cr samples in both θ and
Rq, and whether the adhesion layer choice had a measurable effect on the results.]


#### Limitations of the analysis method

The electron detector Erf–Gaussian method provides an indirect estimate of θ inferred from the top-down intensity profile. It cannot distinguish between a genuinely sloped sidewall and broadening caused by beam effects at the measurement stage. Independent verification by tilted SEM or FIB-TEM cross-section, as described in Section 5.1, would be required to confirm these results at a traceable level of accuracy.



### concluding 
- where the deliverables achieved?
- what was leant

[→Prev: Fabrication ](Fabrication.md)| [Next: Future works →](FW.md)