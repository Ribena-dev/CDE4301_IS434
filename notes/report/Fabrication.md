## Fabrication
### 3.1 overview of fabrication steps 

 
[INSERT animated top-down view of fabrication steps]
 
The fabrication process consists of five sequential steps, as illustrated in Figure 3.1:
 
1. **Silicon wafer** — the base substrate on which all subsequent layers are built.
2. **Spin-coated resist** — PMMA is spin-coated onto the wafer surface to the required thickness (Section 3.3).
3. **Lithography and development** — the grid pattern is written by PBW (Section 3.4) and the exposed resist is removed by DI:IPA development, leaving a patterned resist stencil.
4. **Metal deposition** — metal is deposited by PVD into the open trench regions (Section 3.5).
5. **Resist removal (lift-off)** — the remaining PMMA is dissolved in acetone, removing the metal on top of the resist and leaving only the metal features on the substrate.
 
An optional adhesion layer may be deposited directly onto the silicon wafer prior to spin coating. This intermediate layer serves to improve resist adhesion to the substrate, reduce internal film stress where lattice mismatch between the deposited metal and silicon is large, and improve electrical conductivity or imaging contrast of the final standard

### 3.2 simulations of P-beam in PMMA 

[INSERTspread through PMMA 1um]

[INSERT SRIM range distribution plot — ion count vs depth for 2 MeV protons in PMMA]
 
[INSERT SRIM lateral straggle plot — σ(z) vs depth z for 2 MeV protons in PMMA]
 
SRIM Monte Carlo simulations were used to characterize the behavior of 2 MeV protons in PMMA and to predict the theoretical sidewall angle of the fabricated features. Two outputs are of interest: the depth distribution (Bragg peak), which confirms the feature height achievable at a given beam energy, and the lateral straggle σ(z), which governs edge sharpness as a function of depth.
 
#### Theoretical Sidewall Angle
 
The lateral straggle σ(z) from SRIM gives the standard deviation of the beam's lateral spread at depth z. The edge transition width at that depth is related to σ by:
 
$$ f(z) = 2\sqrt{2\ln 2} \cdot \sigma(z) \approx 2.355\,\sigma(z) $$
 
where f is the FWHM of the dose profile across the feature edge — the same parameter extracted from SEM measurements in Section 2.5.1. The theoretical sidewall angle at the full feature depth h is then:
 
$$ \theta = 90° - \arctan\!\left(\frac{f(h)}{h}\right) = 90° - \arctan\!\left(\frac{2.355\,\sigma(h)}{h}\right) $$

### 3.3 spin coating the waver and development
#### spin coating


Film thickness is governed by two parameters: the concentration (viscosity) of the resist solution and the spin speed [1]. Higher spin speeds and lower concentrations produce thinner films, following an approximate inverse power-law relationship.
 
PMMA is available in two standard molecular weights — 495K and 950K — each supplied at multiple concentrations in anisole (e.g. A2, A4, A6 for 2%, 4%, 6% solids by weight) [1][2]. Higher molecular weight resist is more viscous at the same concentration and produces a slightly thicker film at a given spin speed. The choice of molecular weight and concentration together determine the accessible thickness range:

| Grade | MW | Concentration | Thickness range (1000–5000 rpm) |
|---|---|---|---|
| PMMA 495K A4 | 495,000 | 4% in anisole | ~100–400 nm |
| PMMA 950K A4 | 950,000 | 4% in anisole | ~150–500 nm |
| PMMA 950K A6 | 950,000 | 6% in anisole | ~300–900 nm |


#### Pre-back , Post-bake
After spin coating, the wafer is placed on a hotplate for a soft bake, typically at 180 °C for 60–90 seconds [1] [3]. The pre-bake serves two purposes: it drives off residual solvent (anisole) from the film, which would otherwise cause the resist to remain tacky and deform during handling; and it densifies and hardens the film, improving adhesion to the substrate and reducing unwanted swelling during development. Baking above ~125 °C is avoided as PMMA begins to flow and reflow at elevated temperatures, rounding the resist edges [1].

#### Development and lift off
[ insert video of development]
[insert images of PMMA developemnt]

Development is performed after PBW exposure and is included here for process continuity. The wafer is immersed in DI water:IPA (7:3) developer, which selectively dissolves the chain-scissioned PMMA in the exposed regions while leaving the unexposed resist intact [1][2]. The sample is then rinsed in fresh IPA and dried with a nitrogen gun to stop development. Following metal deposition, the remaining PMMA is removed by immersion in acetone, lifting off the metal on top of the resist and leaving only the metal deposited directly onto the silicon substrate.

### 3.4 P-beam structure
[insert accelerator sruture from quaresh and the lens setup (beam line optics) ]
The PBW facility at CIBA is built around a 3.5 MV Singletron accelerator (HVEE) which generates a focused MeV proton beam for lithography [4] [5]. Protons are produced from hydrogen gas, accelerated to the required energy, and filtered by a 90° analysing magnet before being directed to the PBW end station via a switching magnet. Blanking plates deflect the beam off-axis to control dose delivery during patterning [4].
 
Before focusing, the beam passes through two apertures. The objective aperture (8 × 4 µm²) defines the virtual source size, while the collimator aperture (30 × 30 µm²) reduces angular divergence entering the lenses, giving a beam half-divergence of approximately 3 µrad [4].

Focusing is achieved by a spaced Oxford triplet of magnetic quadrupole lenses in a converging-diverging-converging (CDC) configuration. A single quadrupole focuses in one plane and defocuses in the other; the triplet arrangement produces a symmetric spot focus. With an object-to-lens distance of 7.5 m and image distance of 30 mm, the system achieves a demagnification of 857 × 130, yielding a minimum spot size of 9.3 × 32 nm² [4]. Chromatic aberration — from the finite energy spread of the accelerator — is the dominant limit on spot size, requiring ~10 ppm accelerator stability for sub-10 nm resolution [4]. Before writing, the beam is focused by scanning across a free-standing resolution standard. The transmitted or secondary electron signal produces a complementary error function profile, which is fitted to extract the beam FWHM (Section 2.5). Once focused, a writing file is loaded and the beam is rastered over the resist using electrostatic scanners combined with stage movement for
larger fields [5].

<iframe 
  src="/notes/report/scripts/beam_geo.html" 
  width="100%" 
  height="580px" 
  style="border:none; border-radius:6px;">
</iframe>
  
| Parameter | Value |
|---|---|
| Accelerator | 3.5 MV Singletron (HVEE) |
| Beam energy | 2 MeV protons |
| Objective aperture | 8 × 4 µm² |
| Collimator aperture | 30 × 30 µm² |
| Beam half-divergence | ~3 µrad |
| Lens configuration | Spaced Oxford triplet (CDC) |
| Object-to-lens distance | 7.5 m |
| Image distance | 30 mm |
| Demagnification (X) | 857 |
| Demagnification (Y) | 130 |
| Minimum spot size | 9.3 × 32 nm² |
| Quadrupole power supply resolution | 2 ppm (Bruker) |


### 3.5 Metal deposition characteristics
| Material | Deposition technique | Melting point | conductivity | lattice mismatch |Reasoning| 
| Au | Magnetron sputtering| | | | |




### 3.6 Fabricated samples parameters
| Samples | Composition | Height(nm) | Theoretical sidewall angle | 
| 1 | Au on Cr| | |


<div class="references">

### Reference

[4] S. Raman, Y. Yao, and J. A. van Kan, "Automatic beam focusing in
    the 2nd generation PBW line at sub-10 nm line resolution," Nuclear
    Instruments and Methods in Physics Research Section B, vol. 348,
    pp. 22–26, 2015. DOI: 10.1016/j.nimb.2014.12.066
 
[5] J. A. van Kan, P. Malar et al., "Proton beam writing nanoprobe
    facility design and first test results," Nuclear Instruments and
    Methods in Physics Research Section A, 2011.
    DOI: 10.1016/j.nima.2010.12.011

<ol class="ref-list">
  <li>Microchem / Kayaku Advanced Materials, "PMMA Data Sheet," 2019. Available: <a href="https://kayakuam.com/wp-content/uploads/2019/09/PMMA_Data_Sheet.pdf">kayakuam.com</a></li>
  <li>J. A. van Kan et al., "Resist materials for proton beam writing: a review," <em>Applied Surface Science</em>, 2014. DOI: <a href="https://doi.org/10.1016/j.apsusc.2014.04.147">10.1016/j.apsusc.2014.04.147</a></li>
  <li>University of Chicago Pritzker Nanofab, "NANO 495 PMMA process," 2024. Available: <a href="https://pnf.uchicago.edu/process/detail/950pmma-a4/">pnf.uchicago.edu</a></li>
</ol>
</div>

