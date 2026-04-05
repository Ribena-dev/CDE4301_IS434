---
layout: default
title: "Fabrication"
---

## 2. Fabrication

The grid resolution standard is produced using a lift-off sequence — the most widely adopted approach for producing patterned metallic structures in nanofabrication and the established procedure at CIBA. The process is organised into five sequential steps, each discussed below together with the material and technique decisions that govern that step.

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/lit_fab.png" alt="Fabrication process overview" style="max-width: 680px; width: 100%;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.1:</strong> Lift-off fabrication process for the grid resolution standard.
  </figcaption>
</figure>

---

### 2.1 Spin coating

#### 2.1.1 Resist selection

Resists are radiation-sensitive materials that can be coated onto substrates and locally modified to yield desired patterns. Two of the most widely used high-resolution resists in direct-write nanofabrication are PMMA and HSQ, both of which are compatible with proton-beam writing at sub-100 nm dimensions [10].

**PMMA** (poly(methyl methacrylate)) is a positive resist: ionising radiation breaks the polymer backbone at the carbon–carbonyl bond via chain scission, reducing the molecular weight of the exposed regions and increasing their solubility in a developer such as DI:IPA. The unexposed PMMA is retained as the resist stencil [11][12].

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/PMMA_repeating_unit.svg.png" alt="PMMA repeating unit" style="max-width: 260px; width: 100%;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.2:</strong> PMMA repeating unit.
  </figcaption>
</figure>

**HSQ** (hydrogen silsesquioxane) is a negative resist: radiation crosslinks the cage-like Si–O structure into a dense network that is insoluble in TMAH developer. The unexposed regions dissolve, leaving the crosslinked HSQ as the patterned feature [10][13].

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/Silsesquioxane_T8_Cube.png" alt="HSQ repeating unit" style="max-width: 200px; width: 100%;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.3:</strong> HSQ cage structure.
  </figcaption>
</figure>

**PMMA is selected** for this project because it is the only option compatible with lift-off. As a positive resist, PMMA produces an undercut profile during development: the exposed region beneath the surface is slightly wider than the opening at the top, allowing deposited metal to sit proud of the resist walls and separate cleanly when the resist is dissolved. HSQ, as a negative resist, produces an overcut profile — the feature is wider at the top than at the base — which traps metal against the resist walls and prevents clean lift-off [13].

#### 2.1.2 Spin coating parameters

Film thickness is governed by the resist concentration (viscosity) and the spin speed, following an approximate inverse power-law relationship. Higher spin speeds and lower concentrations produce thinner films [1].

PMMA is available in two standard molecular weights — 495K and 950K — each supplied at multiple concentrations in anisole (A2, A4, A6 for 2%, 4%, 6% solids by weight). Higher molecular weight resist is more viscous at the same concentration and produces thicker films at a given spin speed [1][2].

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/pmma_495k_mid.png" alt="PMMA 495K spin curve medium" width="280" style="margin: 5px;">
  <img src="images/pmma_495k_thin.png" alt="PMMA 495K spin curve thin" width="280" style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.4:</strong> Spin curves for PMMA 495K at medium (left) and low (right) concentrations. Higher spin speed produces thinner films.
  </figcaption>
</figure>

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/pmma950k.png" alt="PMMA 950K spin curve" width="280" style="margin: 5px;">
  <img src="images/pmma950k_thinrange.png" alt="PMMA 950K thin range" width="280" style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.5:</strong> Spin curves for PMMA 950K at standard (left) and dilute (right) concentrations.
  </figcaption>
</figure>

A key design constraint is that the PMMA thickness must be at least five times greater than the intended metal deposition thickness. This ratio ensures sufficient structural integrity of the resist walls and prevents mushrooming — where excess metal forms a cap over the resist that blocks lift-off. Excessively thick PMMA, however, increases the aspect ratio of the trench, risking wall collapse and allowing greater lateral beam straggle to accumulate with depth (Section 2.2.2), which degrades the sidewall angle. The optimal thickness balances these competing constraints against the spin curves above.

#### 2.1.3 Pre-bake

After spin coating, the wafer is placed on a hotplate at 180 °C for 60–90 seconds [1][3]. The pre-bake drives off residual anisole solvent, which would otherwise leave the film tacky and prone to deformation during handling, and densifies the film to improve adhesion to the substrate. Baking above ~125 °C is avoided as PMMA begins to flow and round its edges at elevated temperatures [1].

---

### 2.2 Lithography

#### 2.2.1 Lithography method selection

The lithographic technique governs sidewall verticality, feature resolution, and proximity effects — the three parameters most critical to the performance of a resolution standard. Three primary techniques were evaluated: electron-beam lithography (EBL), focused ion beam (FIB) lithography, and proton-beam writing (PBW).

**EBL** uses a focused Gaussian electron beam to expose resist at resolutions of 10 nm or below [1]. Its fundamental limitation is the proximity effect: high-energy secondary electrons scatter laterally from the primary beam track, exposing resist beyond the intended boundaries. This effect worsens with increasing depth, making consistently vertical sidewalls across a full two-dimensional grid geometry difficult to guarantee [1][2].

**FIB** uses heavy ions, most commonly Ga⁺ at 50 keV, to ablate material directly. It is a subtractive process unsuited to the lift-off sequence used here. The shallow penetration of the Ga beam limits achievable depth, and gallium implantation into the substrate contaminates the metal features and degrades backscatter contrast [3][4].

**PBW** uses a focused MeV proton beam scanned over the resist [4][5]. Protons are approximately 1,800 times more massive than electrons, producing two critical advantages: near-linear trajectories with minimal lateral deflection even at significant depths, and low-energy secondary electrons (typically below 100 eV) with a very short range that modify resist only within a few nanometres of the beam track. The result is negligible proximity effect and well-defined, near-vertical sidewalls [4][5][6].

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/Comparison-between-p-beam-writing-FIB-and-e-beam-writing-This-figure-shows.png"
       alt="EBL vs FIB vs PBW beam spread comparison" style="max-width: 600px; width: 100%;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.6:</strong> Depth penetration and lateral spread comparison between EBL, FIB, and PBW. PBW produces the most linear trajectory with minimal lateral straggle.
  </figcaption>
</figure>

UV lithography is excluded as its resolution is diffraction-limited and unsuitable for the sub-micron grid features required. **PBW is selected** for this project as the only technique capable of producing a metallic grid standard via lift-off with consistent sidewall angles approaching 90° across a full two-dimensional geometry.

#### 2.2.2 SRIM simulations

SRIM Monte Carlo simulations were used to characterise the behaviour of 2 MeV protons in PMMA and to predict the theoretical sidewall angle of the fabricated features. Two outputs are of interest: the Bragg peak depth, which confirms the feature height achievable at a given beam energy; and the lateral straggle σ(z), which governs edge sharpness as a function of depth.

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/srim_trajectories.png" alt="SRIM proton trajectories in PMMA" style="max-width: 680px; width: 100%;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.7:</strong> SRIM Monte Carlo simulation of 2 MeV proton trajectories in 1 µm PMMA. Left: side view (X–Y) showing near-straight ion paths. Right: exit spread (Y–Z) showing the radial distribution at the resist base.
  </figcaption>
</figure>

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/srim_lateral_straggle.png" alt="SRIM lateral straggle vs depth" style="max-width: 680px; width: 100%;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.8:</strong> Lateral straggle σ<sub>r</sub> as a function of depth for 2 MeV protons in PMMA. The grey dashed curve shows raw SRIM data including nuclear scatter outliers. The teal curve shows cleaned data after IQR ×3 outlier removal, giving a true straggle of 0.81 nm at 1 µm — well below the 3 nm target (red dotted line).
  </figcaption>
</figure>

The lateral straggle σ(z) from SRIM gives the standard deviation of the beam's lateral position at depth z. The edge transition width is related to σ by the FWHM conversion:

$$f(z) = 2\sqrt{2\ln 2}\cdot\sigma(z) \approx 2.355\,\sigma(z)$$

The theoretical sidewall angle at feature depth h is then:

$$\theta = 90° - \arctan\!\left(\frac{f(h)}{h}\right) = 90° - \arctan\!\left(\frac{2.355\,\sigma(h)}{h}\right)$$

At h = 1000 nm, the cleaned SRIM straggle of 0.81 nm gives f = 1.91 nm and θ = 89.9° — confirming that PBW is theoretically capable of meeting the ≥89.4° deliverable.

#### 2.2.3 PBW setup at CIBA

PBW was carried out at CIBA using a 3.5 MV High Voltage Engineering Europa (HVEE) Singletron accelerator coupled to a dedicated proton-beam writing end station. The system uses a set of magnetic quadrupole lenses to demagnify the beam from an object aperture, producing a sub-micron focused spot at the resist surface.

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/beam_line.png" alt="CIBA beam line optics" style="max-width: 600px; width: 100%;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.9:</strong> Beam line optics at CIBA.
  </figcaption>
</figure>

The beam is deflected electrostatically to raster-scan the desired pattern over the resist surface. A beam blanker deflects the beam off-axis between pattern elements to prevent unwanted dose. For this project, a 2 MeV proton beam with a spot size of 9.3 × 32 nm² was used to write the 100 µm × 100 µm grid pattern. The focal plane accuracy is ±1 µm.

#### 2.2.4 Dose and energy

Dose in proton-beam writing refers to the total charge delivered per unit area of resist, expressed in nC/mm². It is the product of beam current, dwell time per pixel, and the inverse of the pixel area — physically, the number of protons that have passed through each unit area of resist surface.

Energy determines the depth at which the protons stop. At 2 MeV, protons penetrate approximately 60 µm into PMMA, well beyond the 1 µm resist thickness used here, ensuring the Bragg peak lies in the silicon substrate rather than within the resist. For the 1 µm resist thickness used in this project, the protons traverse the full resist depth with near-uniform energy loss, depositing dose uniformly from surface to base.

The clearing dose threshold for PMMA with 2 MeV protons is approximately 50–75 nC/mm². Below this, chain scission density is insufficient for the developer to dissolve the exposed material. Above approximately 150–280 nC/mm², overdose begins to widen the trench beyond the written pattern. A dose test grid spanning 75–175 nC/mm² was written to identify the optimal dose for each sample configuration.

---

### 2.3 Development and lift-off

#### 2.3.1 Development

Following PBW exposure, the wafer is immersed in a DI water : isopropanol (IPA) 7:3 developer for 60 seconds, then rinsed in DI water and dried under nitrogen. The developer selectively dissolves the chain-scissioned PMMA in the exposed regions, leaving the unexposed resist as a patterned stencil with open trenches where the grid features will be formed.

#### 2.3.2 Lift-off

After metal deposition (Section 2.4), the wafer is immersed in acetone to dissolve the remaining PMMA. The metal film on top of the resist is mechanically separated and removed along with the dissolving polymer, leaving only the metal that was deposited directly onto the silicon substrate within the developed trenches. The result is the patterned metallic grid feature array.

---

### 2.4 Metal deposition

#### 2.4.1 Deposition method

Metal deposition is carried out using physical vapour deposition (PVD), in which a solid source material is vaporised under high vacuum and condensed onto the substrate as a thin film [13][14].

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/PVD_schmatic.png" alt="PVD schematic" style="max-width: 480px; width: 100%;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.10:</strong> PVD process schematic.
  </figcaption>
</figure>

Three PVD techniques were evaluated for this project.

**Magnetron sputtering** uses argon plasma to bombard a target, ejecting atoms with energies of 1–10 eV. The diffuse angular flux promotes three-dimensional island growth in the deposited film, producing a granular surface texture. For lift-off applications, the broad angular distribution of sputtered atoms risks coating the resist sidewalls, which can bridge the trench opening and prevent clean separation [15].

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/RF_sch.png" alt="RF sputtering schematic" style="max-width: 400px; width: 100%;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.11:</strong> RF magnetron sputtering schematic.
  </figcaption>
</figure>

**Electron-beam (e-beam) evaporation** focuses a high-voltage electron beam onto a target in a water-cooled crucible, causing it to sublimate and produce a directional vapour flux at lower energies (0.1–1 eV). The line-of-sight transport geometry promotes flatter, more conformal deposition and is better suited to lift-off than sputtering. The process can handle refractory metals with melting points up to ~2,800 °C.

**Filtered cathodic vacuum arc (FCVA)** strikes a high-current arc on a graphite cathode, generating a carbon plasma directed through a magnetic filter that removes macroparticles. This produces dense, smooth diamond-like carbon (DLC) films with a tunable sp²/sp³ ratio at room temperature, without requiring a precursor gas.

#### 2.4.2 Material selection

Four criteria were used to select the metal layer materials for this project. The material must be compatible with PMMA lift-off (low substrate temperature during deposition), provide high electron backscatter contrast for SEM calibration use (high atomic number Z preferred), be chemically stable against oxidation during storage and repeated use, and produce a smooth surface film with Rq below 3 nm as required by the Névot–Croce analysis in Section 2.6.2.

The Névot–Croce factor describes the attenuation of the specular reflected signal from a rough surface:

$$R = R_0 \exp\!\left(-\left(\frac{4\pi\,R_q\,\cos\theta}{\lambda}\right)^2\right)$$

At q_z = 1.0 nm⁻¹, signal retention falls below 80% for Rq > 3 nm, establishing the surface roughness target [18].

Based on these criteria the following materials were selected and tested across the seven sample configurations described in Section 2.4.3:

| Material | Z | Deposition method | Role |
|---|---|---|---|
| Gold (Au) | 79 | Magnetron sputtering | Primary grid metal, high contrast |
| Palladium (Pd) | 46 | E-beam evaporation | Alternative grid metal, smooth film |
| Diamond-like carbon (DLC) | 6 | FCVA | Low-Z contrast reference layer |
| Chromium (Cr) | 24 | E-beam evaporation | Adhesion layer (for Au) |
| Titanium (Ti) | 22 | E-beam evaporation | Adhesion layer (for Pd) |

#### 2.4.3 Sample configurations

Seven sample configurations were fabricated to compare the effect of metal stack choice on sidewall angle, surface roughness, and electron contrast.

| Sample | Substrate | Adhesion layer | Primary metal | Top layer |
|---|---|---|---|---|
| 1 — Au/Cr/Si | Si | Cr 2 nm | Au 40 nm | — |
| 2 — DLC/Si | Si | — | — | DLC |
| 3 — DLC/Pd/Si | Si | — | Pd | DLC |
| 4 — Au/Pd/Si | Si | — | Pd | Au 40 nm |
| 5 — Pd/Cr/Si | Si | Cr 2 nm | Pd 40 nm | — |
| 6 — DLC/Pd/Ti/Si | Si | Ti 2 nm | Pd | DLC |
| 7 — Pd/Ti/Si | Si | Ti 2 nm | Pd 40 nm | — |

Note: Sample 7 (Pd/Ti) was fabricated but could not be characterised within the project timeline due to equipment downtime. Results for all other samples are presented in Chapter 3.

---

### 2.5 Analysis methods

#### 2.5.1 AFM surface roughness

Surface roughness was measured by atomic force microscopy (AFM) in tapping mode. Three standard parameters are reported for each profile of N height points y_i, after subtracting the mean height to remove scan tilt:

$$R_q = \sqrt{\frac{1}{N}\sum_{i=1}^{N}y_i^2}$$

$$R_a = \frac{1}{N}\sum_{i=1}^{N}|y_i|$$

$$R_z = y_{\max} - y_{\min}$$

Rq is the primary metric for this project as it appears in the Névot–Croce expression above. Ra treats all deviations equally and is reported as a secondary reference. Rz gives the peak-to-valley worst case and is sensitive to isolated spikes or scratches. Two orthogonal profiles were measured for each sample: P1 along the horizontal scan axis and P2 along the vertical scan axis, enabling assessment of surface anisotropy.

#### 2.5.2 Sidewall angle via Erf–Gaussian edge fitting

Sidewall angle θ is extracted from electron detector intensity profiles using a combined error function and Gaussian model [13]:

$$F(x) = A\!\left[1 + \text{Erf}\!\left(\frac{2\sqrt{\ln 2}}{f}(d-x)\right)\right] + B\exp\!\left(-\frac{\ln 16}{f^2}(d-x)^2\right) + C$$

where A is the error function amplitude, B is the Gaussian amplitude representing the secondary electron peak at the sidewall, C is the baseline signal, d is the physical edge position, and f is the FWHM of the edge transition [13].

The sidewall angle is then calculated from f and the known feature height h:

$$\theta = 90° - \arctan\!\left(\frac{f}{h}\right)$$

A sidewall angle of exactly 90° corresponds to a perfectly vertical wall. Shallower angles produce larger f values and lower θ. The target of ≥89.4° corresponds to a maximum FWHM of f ≤ h × tan(0.6°).

---

### References

1. Kayaku Advanced Materials, "PMMA technical datasheet," 2022.
2. K. Yamazaki, "Electron beam direct writing," in *Nanofabrication: Fundamentals and Applications*, A. A. Tseng, Ed., World Scientific, 2008.
3. F. Watt, M. B. H. Breese, A. A. Bettiol, and J. A. van Kan, "Proton beam writing," *Materials Today*, vol. 10, no. 6, pp. 20–29, 2007.
4. G. Winkler et al., "FIB roadmap for focused ion beam technology," 2020.
5. J. A. van Kan et al., "Proton beam writing," *Nano Letters*, 2006.
6. A. A. Rajendran et al., "Sub-10 nm proton beam writing," CIBA, NUS, 2011.
7. A. A. Bettiol, S. V. Rao, E. J. Teo, J. A. van Kan, and F. Watt, "Sidewall quality in proton beam writing," *Nuclear Instruments and Methods B*, vol. 258, pp. 302–306, 2007. DOI: 10.1016/j.nimb.2007.02.065
8. J. A. Gierak, "Ga implantation in FIB processing," 2009.
9. D. Mack, *Fundamental Principles of Optical Lithography*. Wiley, 2011.
10. K. A. Tseng, *Electron Beam Lithography in Nanofabrication*. World Scientific, 2008.
11. Kayaku Advanced Materials, "PMMA A-series datasheet," 2022.
12. O. Müller et al., "PMMA thermal flow effects," *Microelectronic Engineering*, 2001.
13. NIST, "CD-AFM metrology reference," 2016. Available: nist.gov
14. B. Freund and S. Suresh, *Thin Film Materials*. Cambridge University Press, 2003.
15. D. Frey et al., *Handbook of Thin Film Technology*. Springer, 2015.
16. Kayaku, PMMA datasheet, 2022.
17. van Kan et al., "PBW resist review," *Microsystem Technologies*, 2008.
18. L. Névot and P. Croce, "Roughness attenuation factor," *Revue de Physique Appliquée*, 1980.

[← Introduction](Introduction.html) · [Results and Analysis →](fna.html)