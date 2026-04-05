# Fabricating resolution standards using Proton beam lithography

# Acknowledgements


# Abstract 


# Introduction


<figure style="text-align: center; margin: 20px 0;">
  <img src="images/Scaling-of-transistor-size-physical-gate-length-L-g-to-sustain-Moores-Law.png" alt="moores law tranistor gate scalling" >
  
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 1.1:</strong> Scaling of transistor size physical-gate length to-sustain Moore's Law
  </figcaption>
</figure>

Moore's Law, which predicts the doubling of transistor density approximately every two years which has driven semiconductor feature sizes  from the micrometre range in the 1970s to sub-2 nm nodes  (thinner that a human DNA) in commercial production today [1]. These features are often fabricated through complex proprietary steps , that are outside the scope of this report. However they can be brutishly summarized to the the following steps  of deposition, patterning (ex. lithography), and etching on a polished silicon wafer.

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/sem_2nm.png" alt="SEM of individual transistor on IBM's chip" >
  
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 1.2:</strong> scanning electron microscope image of individual transistors, each measuring 2 nanometers wide
  </figcaption>
</figure>
ref: https://newatlas.com/computers/ibm-2-nm-chips-transistors/

This relentless miniaturisation has rendered conventional optical microscopy impractical for surface characterisation ,the wavelength of visible light (380–700 nm) is far greater than the dimensions of current transistor features [2]. This raises a fundamental question: how can such structures be characterised with the precision required for manufacturing?

As shown above, characterizing instruments such as [scanning electron microscopes (SEM)](https://microbenotes.com/scanning-electron-microscope-sem/), [critical dimension atomic force microscopes (CD-AFM)](https://www.nist.gov/programs-projects/atomic-force-microscopy), [transmission electron microscopes (TEM)](https://microbenotes.com/transmission-electron-microscope-tem/), and [extreme ultraviolet (EUV) scatterometry systems](https://www.nist.gov/programs-projects/euv-scatterometry), are being pushed to the limits of accuracy to validate such structures. 

However, the accuracy of measurements from any such instrument depends entirely on the quality of its calibration [3] ,which is where resolution and calibration standards become essential.

## 1.1 Overview of Resolution Standards

There are many kinds of resolution standards/calibration standards, below are the examples of tin spheres and a fine nano copper mesh 

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/tin_nm.png" alt="Tin sphere resolution standards" width="280" style="margin: 5px;">
  <img src="images/grid_nm.png" alt="Grid resolution standards" width="280" style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 1.2:</strong> Common resolution standards: tin spheres (left) and nano-grids (right)
  </figcaption>
</figure>

As expected calibrating such complex machines would require a complex degree of steps with various kinds of standards, for instance tin spheres are more commonly used for used for exposure and coverage testing, but are not applicable to CD-AFM calibration. Unlike the resolution grids, these can be used in calibration of all the above mentioned machines.

## 1.2 Problem statement

How are current resolution grids made?

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/lit_fab.png" alt="resolution fabrication overview"  style="margin: 5px;">
  
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 1.3:</strong> Fabrication process for resolution standard overview side view
  </figcaption>
</figure>

The above is a rather simplified, overview of the fabrication process. (More details will be given later)
Heres the issue, most commercial grids are made using electron beam lithography (EBL)(step 2)

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/e_beam.png" alt="resolution fabrication overview"  style="margin: 5px;">
  
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 1.4:</strong> simplified EBL on positive resist 
  </figcaption>
</figure>

When we zoom into the E-Beam penetrating the resist material we get this: 

<iframe src ="scripts/ebeam_vs_pbeam_lateral_spread.html">
</iframe>