# Fabrication and Characterization of 3D Resolution Standards Using Proton Beam Lithography

**Student:** Devinaa Kumeresh  
**Student ID:** A0266490X  
**Project Code:** IS-434   
**Semester:** AY2025 Semester 1 Interim Report  
*Word Count:~2800*

## Acknowledgments
I want to express thanks to everyone who has helped me in the first steps of this project 

I want to specifically thank the following individuals: 
My supervisor, Dr. Jeroen Anton, for his invaluable guidance and support throughout this project. 

Dr. Tan Chuan Jia and Ming Feng Yee for their technical expertise and assistance.

My fellow research students for their constructive feedback and insights during our group meetings, which have greatly contributed to the progress of this work.


## Table of Contents

### [1. Introduction](index.md)
- [1.1 Resolution Standards and Their Importance](index.md#11-resolution-standards-and-their-importance)
- [1.2 Technological Limitations and Research Motivation](index.md#12-technological-limitations-and-research-motivation)

### [2. Research Objectives and Design Significance](objectives.md)
- [2.1 Project Aim](objectives.md#21-project-aim)
- [2.2 Design Significance](objectives.md#22-design-significance)
- [2.3 Fabrication Process Overview](objectives.md#23-fabrication-process-overview)

### [3. Materials Selection and Considerations](materials.md)
- [3.1 PMMA Resist Selection](materials.md#31-pmma-resist-selection)
- [3.2 Metal Selection and Challenges](materials.md#32-metal-selection-and-challenges)
- [3.3 Beam Writing Strategy Investigation](materials.md#33-beam-writing-strategy-investigation)
- [3.4 Focal Plane Variation](materials.md#34-focal-plane-variation)

### [4. Fabrication Process Breakdown](fabrication.md)
- [4.1 Monte Carlo Simulations](fabrication.md#41-monte-carlo-simulations)
- [4.2 Metal Seed Layer Sputtering](fabrication.md#42-metal-seed-layer-sputtering)
- [4.3 Spin Coating](fabrication.md#43-spin-coating)
- [4.4 Proton Beam Writing](fabrication.md#43-spin-coating)
- [4.5 Diamond-Like Carbon Deposition](fabrication.md#44-proton-beam-writing)
- [4.6 PMMA Development and Acetone Lift-Off](fabrication.md#46-pmma-development-and-acetone-lift-off)

### [5. Results and analysis](analysis_results.md)
- [5.1 Results and Observations](analysis_results.md#51-results-and-observations)
- [5.2 Edge Analysis and FWHM Methodology](analysis_results.md#52-edge-analysis-and-fwhm-methodology)
- [5.3 DLC and Gold Coating Contrast](analysis_results.md#53-dlc-and-gold-coating-contrast)

### [6. Next Steps](next_steps.md)

### [7. References](ref.md)

### [Annex A: Software Overview](A.md)

### [Annex B: Stopping Graph for Different Metals](B.md)
---

## 1. Introduction

### 1.1 Resolution Standards and Their Importance

Resolution standards are physical calibration artifacts featuring precisely engineered nanostructures that serve as reference specimens for microscopy systems [1]. These standards typically consist of periodic patterns such as grids, gratings, or spherical particles with well-characterized dimensions at the nanometer scale, providing functionality for magnification calibration, distortion correction, resolution testing, and astigmatism correction.
<figure style="text-align: center; margin: 20px 0;">
  <img src="images/tin_nm.png" alt="Tin sphere resolution standards" width="280" style="margin: 5px;">
  <img src="images/grid_nm.png" alt="Grid resolution standards" width="280" style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 1.1:</strong> Common resolution standards: tin spheres (left) and nano-grids (right)
  </figcaption>
</figure>
The critical role of resolution standards in nanotechnology and materials science cannot be overstated. As semiconductor fabrication pushes toward sub-3nm technology nodes and biological imaging demands molecular-level resolution, accurate calibration standards become paramount [2]. Without standardized calibration, measurements from different microscopes or laboratories may vary significantly. Resolution standards provide a common reference point, ensuring measurement consistency across facilities and enabling reproducibility and comparability of data across research institutions and industrial facilities [3].

To note, the above examples show the most common resolution standards nano tin spheres and nano grids, neither one is better than the other. They both serve diffrent use cases. Grids are typically used for checking magnification and  distortion, while tin spheres with their varied spacing is used for exposure and light testing. The following project focus on the lithography method of fabricating grid resolution standards. 

### 1.2 Technological Limitations and Research Motivation

Current resolution standards are predominantly designed for surface measurements. As nanofabrication advances into three-dimensional fabrication, there is increasing requirement for subsurface imaging capabilities [4], particularly in semiconductor chip manufacturing with multiple stacked layers and protein imaging for biological research where three-dimensional structures extend through substantial depths [5].

Current grid resolution standards fabricated using electron beam lithography suffer from fundamental limitations when creating deep structures. High-energy electrons scatter from nuclei, creating secondary electrons that expose resist far from the intended beam position. This proximity effect worsens with depth, causing blurred features, edge definition loss, and non-homogeneous dose distribution. Electron beam broadening below the surface makes vertical sidewalls impossible at depths beyond a few hundred nanometers [6].
<figure style="text-align: center; margin: 20px 0;">
  <img src="images/p_e_comparison.png" alt="Proton and electron scattering comparison" width="350">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 1.2:</strong> Lateral scattering comparison between electron and proton beams
  </figcaption>
</figure>
In contrast, protons are approximately 1800 times heavier than electrons (mp/me ≈ 1800), resulting in minimal proximity effects due to reduced momentum transfer during collisions. Protons maintain linear trajectories with minimal lateral spreading even at significant depths, enabling penetration exceeding 100 micrometers with maintained resolution [7]. This fundamental physics creates ideal conditions for fabricating three-dimensional resolution standards with straight edges and accurate dimensions throughout their depth.

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/beam_crossection.png" alt="Electron vs proton beam cross-section comparison" width="500">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 1.3 :</strong> Cross-sectional profiles comparing electron-beam and proton-beam lithography at diffrent thickness
  </figcaption>
</figure>

This diagram compares sidewall profiles produced by electron-beam versus proton-beam lithography. Electron-beam lithography


[Next: Research Objectives →](objectives.md)

