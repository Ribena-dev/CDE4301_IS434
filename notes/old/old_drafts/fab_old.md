## 4. Fabrication Process Breakdown

### 4.1 Monte Carlo Simulations

Simulations of proton beam at 2 MeV through 100 μm and 40 μm PMMA show depth and scattering characteristics. 
<figure style="text-align: center; margin: 20px 0;">
  <img src="images/100um.png" alt="Simulation of 2MeV through 100um PMMA" width="350" style="margin: 5px;">
  <img src="images/40um.png" alt="Simulation of 2MeV through 40um PMMA" width="350" style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 4.1:</strong> Monte Carlo simulations of 2 MeV proton beam through 100 μm (left) and 40 μm (right) PMMA
  </figcaption>
</figure>

Below 15 μm depth, beam spread barely exceeds 0.1 μm, validating that proton beam has potential to create perpendicular vertical sidewalls with homogeneous dosage distribution [13].

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


Unsmooth structures cause final surface roughness, shadowing effects potentially preventing proper metal adhesion during plating, creating areas with more or less coating, and similar issues with PMMA resist coating potentially leaving residues.

### 4.3 Spin Coating

Spin coating parameters derive from previous work. PMMA thickness depends on viscosity and spin coater RPM. Positive resist PMMA is used in high-resolution applications for sharp edge definitions. Current sample uses 1 μm thick PMMA at 4000 RPM [15] [16].

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/spin.png" alt="Spin coating process" width="350" style="margin: 5px;">
  <img src="images/post_spin.png" alt="Post spin coating" width="350" style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 4.5:</strong> Spin coating process (left) and coated sample (right)
  </figcaption>
</figure>

### 4.4 Proton Beam Writing

Writing design employed high dosage with single pass for this sample. Initial sample tests dosage range because beam is calibrated for 0.75 MeV, less than simulated example. Based on interpolation, minimum development dosage is 100 nC/mm². Higher dosages were tested to observe substantial effects on development or fabrication. More information on the diffrent stopping power against beam energy can be found in annex B.

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/dosage.png" alt="Dosage pattern diagram" width="250">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 4.6:</strong> Proton beam dosage test pattern in nC/mm² (each square represents 100 μm × 100 μm grid)
  </figcaption>
</figure>

The diagram above is in nC/mm², each square represent a full 100um by 100um grid

### 4.5 Diamond-Like Carbon Sputtering

Ideally, metal plating would be used. Previous work used nickel or copper plating, but machine maintenance necessitated trying DLC sputtering. Unlike metal plating, sputtering covers entire surface and is harder to direct. Technique relies heavily on vacuum systems, with each metal having different ideal settings, meaning surface smoothness depends on both metal and sputtering settings.

DLC is non-conducting material. Since gold base layer is conductive, having DLC on top builds grid structure. When using SEM, stark contrast should exist between DLC layer electron backscatter and gold layer electron backscatter, providing higher contrast. This depends on DLC thickness; if too thin, electrons pass through, reducing contrast ratio.

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/dlc_afm.png" alt="DLC surface AFM" width="450">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 4.7:</strong> AFM image showing smooth DLC surface deposition
  </figcaption>
</figure>

From the above AFM of DLC, we can tell it has a much smoother surface deposition compared to Gold. This is beneficial, a rougher surface has the potential to cause electrons to backscatter at wild angles, causing the resulting SEM image to be fuzzy.

### 4.6 PMMA Development and Acetone Lift-Off

As predicted, dosages below minimum did not develop, evident by darker coloring in PMMA development and nearly or completely absent in acetone lift-off. This likely results from protons not interacting with suitable depth, leaving last PMMA layer non-reacted. Higher dosages showed structures faster; greater intensity has more probability of interactions in resist material.

<figure style="text-align: center; margin: 20px 0;">
  <img src="20250930 PMMA Grid/20250929_5x_grid1_1minDev_1.png" alt="PMMA development 1 min" width="350" style="margin: 5px;">
  <img src="20250930 PMMA Grid/20250929_5x_grid1_10minDev_1.png" alt="PMMA development 10 min" width="350" style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 4.8:</strong> PMMA development at 1 minute (left) and 10 minutes (right)
  </figcaption>
</figure>

At 20 minutes acetone lift-off, some PMMA leftovers remain visible. Removing PMMA cleanly takes time. Ultrasonicator use risks structure resonance frequency matching with ultrasonicator, potentially completely shearing off base layer and destroying structure.
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
