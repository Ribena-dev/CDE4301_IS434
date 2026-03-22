## 4. Fabrication Process Breakdown

### 4.1 Monte Carlo Simulations

To understand how the proton beam would behave in my fabrication process, I carried out Monte Carlo simulations of a 2 MeV proton beam through 100 μm and 40 μm PMMA. These simulations allowed me to visualise both penetration depth and lateral scattering characteristics under conditions relevant to my intended writing depth. 

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/100um.png" alt="Simulation of 2MeV through 100um PMMA" width="350" style="margin: 5px;">
  <img src="images/40um.png" alt="Simulation of 2MeV through 40um PMMA" width="350" style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 4.1:</strong> Monte Carlo simulations of 2 MeV proton beam through 100 μm (left) and 40 μm (right) PMMA
  </figcaption>
</figure>

From the results, I observed that below 15 μm depth the beam spread remained below 0.1 μm. This confirmed that the proton beam could in principle produce nearly vertical sidewalls with a uniform dosage distribution, consistent with expectations reported in [13]

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/spread_depth.png" alt="Beam spread versus depth graph" width="500">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 4.2:</strong> Average beam spread versus depth showing exponential increase from 0.015 μm at 5 μm to 0.6 μm at 40 μm
  </figcaption>
</figure>

### 4.2 Metal Seed Layer Sputtering

Magnetron sputtering deposits thin metal film providing conductive seed for subsequent processing. Since PMMA is insulating, predeposited layer like gold, chromium, or palladium enables electroplating selectively in exposed regions after lithographic development. Additionally, thin metal film as intermediate layer creates chemically responsive interface enabling clean structure separation from substrate for freestanding nanostructure fabrication [14].

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/sputerring.png" alt="Magnetron sputtering diagram" width="450">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 4.3:</strong> Magnetron sputtering process schematic
  </figcaption>
</figure>

Gold provides high conductivity but sputtering technique and gold characteristics lead to lumpy, unsmooth structure. AFM imaging reveals rough surface morphology.

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/au_afm.png" alt="Gold surface AFM" width="450">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 4.4:</strong> AFM image showing rough gold surface morphology
  </figcaption>
</figure>


This roughness became a concern because it could propagate into the final structure, introduce shadowing during plating, and interfere with PMMA coating quality. These artefacts motivated my later decision to try alternative top-layer materials.
### 4.3 Spin Coating

Spin coating parameters derive from previous seniors work. PMMA thickness depends on viscosity and spin coater RPM. Positive resist PMMA is used in high-resolution applications for sharp edge definitions. Current sample uses 1 μm thick PMMA at 4000 RPM [15] [16].

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/spin.png" alt="Spin coating process" width="350" style="margin: 5px;">
  <img src="images/post_spin.png" alt="Post spin coating" width="350" style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 4.5:</strong> Spin coating process (left) and coated sample (right)
  </figcaption>
</figure>

### 4.4 Proton Beam Writing

When writing the dose test patterns, we used a high-dosage single-pass approach. Because the system was calibrated for 0.75 MeV rather than the 2 MeV used in simulation, we first needed to estimate the appropriate dose. Using interpolation, determining that the minimum development dose for my conditions was roughly 100 nC/mm². We then exposed a range of higher doses to study their influence on development and structural definition. (Further details on stopping power calculations are included in Annex B.)

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/dosage.png" alt="Dosage pattern diagram" width="250">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 4.6:</strong> Proton beam dosage test pattern in nC/mm² (each square represents 100 μm × 100 μm grid)
  </figcaption>
</figure>

The diagram above is in nC/mm², each square represent a full 100um by 100um grid

### 4.5 Diamond-Like Carbon Sputtering

Although my original plan involved metal electroplating (as in previous nickel and copper processes), equipment downtime required me to try DLC sputtering instead. Unlike plating, sputtering coats the entire exposed surface, and the result depends heavily on vacuum conditions and the specific sputtered material. I therefore evaluated DLC both as a structural layer and as a contrast-enhancing surface for SEM imaging.

Because DLC is non-conductive, but the underlying seed layer is gold, I expected a clear backscatter contrast. This depended strongly on DLC thickness; overly thin layers allow electron penetration and reduce contrast.

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/dlc_afm.png" alt="DLC surface AFM" width="450">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 4.7:</strong> AFM image showing smooth DLC surface deposition
  </figcaption>
</figure>

From the above AFM of DLC, can tell us it has a much smoother surface deposition compared to Gold. This is beneficial, a rougher surface has the potential to cause electrons to backscatter at wild angles, causing the resulting SEM image to be fuzzy.

### 4.6 PMMA Development and Acetone Lift-Off

During development, I observed that doses below the estimated threshold did not properly develop, evident by darker coloring in PMMA development and nearly or completely absent in acetone lift-off. This likely results from protons not interacting with suitable depth, leaving last PMMA layer non-reacted. Higher dosages showed structures faster; greater intensity has more probability of interactions in resist material.

<figure style="text-align: center; margin: 20px 0;">
  <img src="20250930 PMMA Grid/20250929_5x_grid1_1minDev_1.png" alt="PMMA development 1 min" width="350" style="margin: 5px;">
  <img src="20250930 PMMA Grid/20250929_5x_grid1_10minDev_1.png" alt="PMMA development 10 min" width="350" style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 4.8:</strong> PMMA development at 1 minute (left) and 10 minutes (right)
  </figcaption>
</figure>

In the lift-off step, even after 20 minutes of acetone I still observed PMMA residues. Removing the remaining resist required patience, but I avoided aggressive ultrasonication because the resonance risk could shear off the metal base layer and permanently damage the structures. The effect of ultrasonication is nevertheless shown below for comparison.

<figure style="text-align: center; margin: 20px 0;">
  <img src="20251003 PMMA Acetone/20250929_5x_grid3_1minAcetone_1.png" alt="Acetone lift-off 1 min" width="300" style="margin: 5px;">
  <img src="20251003 PMMA Acetone/20250929_5x_grid3_20minAcetone_1.png" alt="Acetone lift-off 20 min" width="300" style="margin: 5px;">
  <img src="20251003 PMMA Acetone/20250929_5x_grid3_20minAcetone_IPA-DI-Rinse_ult_7min_1.png" alt="Acetone lift-off with ultrasonic" width="300" style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 4.9:</strong> Acetone lift-off at 1 minute (left), 20 minutes (center), and 20 minutes plus 7 minutes ultrasonication (right)
  </figcaption>
</figure>

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/shake.png" alt="actone lift off" width="300" style="margin: 5px;">
  <img src="images/lift.png" alt="actone lift off" width="300" style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 4.10:</strong> Acetone lift-off diagram
  </figcaption>
</figure>


[← Material selection and consideration](objectives.md) | [Next: Results and Analysis →](analysis_results.md)
