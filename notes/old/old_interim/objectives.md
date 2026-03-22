## 2. Research Objectives and Design Significance

### 2.1 Project Aim

This project aims to produce grid resolution standards with straight perpendicular vertical edges for calibrating high-resolution scanning electron microscopy. Fabricated standards will be evaluated using SEM imaging by analyzing backscattered electron intensity profiles, then applying mathematical models to determine full width at half maximum (FWHM) of beam spot size.
<figure style="text-align: center; margin: 20px 0;">
  <img src="images/grid_dia.png" alt="Grid resolution standard structure diagram" width="300">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.1:</strong> Schematic of intended grid resolution standard
  </figcaption>
</figure>

### 2.2 Design Significance

The primary goal is fabricating nanometer-scale SEM resolution standard samples for calibrating high-resolution imaging systems [9]. A key design improvement minimizes measurement errors from non-ideal sample geometry, specifically sloped sidewalls of metal grid features. In high-resolution SEM analysis, geometric deviations lead to inaccurate beam size estimation. Fabricating samples with vertical metal edges reduces artifacts in edge intensity profiles and enhances FWHM-based resolution estimation reliability [10].

When SEM beam spot size is characterized using edge analysis, measured intensity transition width reflects both actual beam diameter and geometric contributions from sample edge roughness or slope. Sloped sidewalls cause gradual backscattered electron signal transitions, artificially broadening apparent beam size. Ensuring near-vertical sidewalls through proton beam lithography minimizes this geometric artifact and enables accurate SEM beam resolution characterization.

### 2.3 Fabrication Process Overview

The process begins with sputtering thin metal seed layer onto silicon substrate for conductivity and potential release layer functionality. PMMA positive resist is spin-coated, followed by proton beam writing to create grid patterns. After development revealing exposed structures, metal deposition covers the entire surface. Finally, acetone lift-off removes PMMA resist and overlying metal, leaving patterned metal grid structures.
<figure style="text-align: center; margin: 20px 0;">
  <img src="images/brief_fab.png" alt="Fabrication process flow diagram" width="1000px" height="60px">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.2:</strong> Overview of fabrication process
  </figcaption>
</figure>

[← Introduction and background](index.md) | [Next: Material selection and considerations →](materials.md)