---
layout: default
title: "Methodology"
---

<!-- <iframe src ="scripts/rn_process_nav.html"
        allowfullscreen="true" 
        width="700px" 
        height="400px"> 
</iframe> -->


## Methodology

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/lit_fab_2.png" alt="resolution fabrication overview"  >
  
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.Y:</strong> Fabrication process for resolution standard overview side view
  </figcaption>
</figure>

This section outlines the materials, techniques, and analytical methods selected to fabricate and characterise the grid resolution standard. The fabrication process follows a standard sequence, which is the most widely adopted approach for producing patterned metal structures in nanofabrication and is the established procedure at CIBA. The process proceeds in four stages:

1. **Spin coating**: a photo resist is spin-coated onto the substrate to form a uniform film of controlled thickness.
2. **Lithography**: the resist is exposed 
3. **Metal deposition**: a metal layer is deposited over the entire surface, filling the developed trenches.
4. **Lift-off**: the remaining resist is dissolved in solvent, removing the metal deposited on top of it and leaving only the patterned grid structures.

To note section will skip a general lithography section as the novelty in section 1 already covers it , instead focusing more on understanding the Proton beam 


### 2.1 Resist 

Resists are radiation-sensitive materials that can be coated onto substrates and locally modified to yield desired patterns. Based on their response to exposure, they are broadly classified as positive or negative resists. Positive resists show an increased dissolution rate in the exposed regions when developed, while negative resists show a decreased dissolution rate ,the exposed regions become
insoluble and are retained after development [9].

Two of the most common and highest-resolution resists used for nano-fabrication are PMMA and HSQ. Both have been demonstrated to be compatible with proton-beam writing (PBW) at sub-100 nm dimensions, and represent the state-of-the-art for
direct-write lithography at CIBA [10].


####  PMMA ,Poly(methyl methacrylate)

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/PMMA_repeating_unit.svg.png" alt="Resolution lithography process" width="280" >
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.Y:</strong> PMMA repeating unit
  </figcaption>
</figure>

PMMA is a long-chain synthetic polymer and one of the most widely used positive resists in nano-fabrication. Its primary advantages include a simple formulation (PMMA dissolved in anisole, a low-toxicity solvent), insensitivity to white
light (λ > 250 nm), a wide range of available film thicknesses through dilution, no shelf-life limitations, no processing delay effects after spin-coating, and straightforward removal after metal deposition via lift-off using acetone to dissolve the pmma[10] [11]. 

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/p_re.png"  width="280" >
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.Y:</strong> PMMA simplified development
  </figcaption>
</figure>

PMMA is a positive resist. When exposed to a proton or electron beam, the incident radiation generates secondary electrons that initiate chain scission ,the breaking of the polymer backbone at the carbon–carbonyl bond. This reduces the molecular weight of the polymer in the exposed regions, increasing their solubility in an organic developer such as MIBK:IPA. The exposed material is dissolved away during development , leaving the unexposed PMMA as the remaining resist pattern [11] [12].

#### HSQ ,Hydrogen Silsesquioxane

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/Silsesquioxane_T8_Cube.png" alt="Resolution lithography process" width="280" >
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.4:</strong> HSQ repeating unit
  </figcaption>
</figure>

HSQ is an inorganic silicon-based resist with the empirical formula [HSiO₃/₂]ₙ.In its as-deposited state it exists as a polyhedral cage of silicon and oxygen atoms, each silicon bearing a single hydrogen substituent. HSQ is a negative resist and has been shown to function as a high-resolution negative-tone e-beam resist, with resolutions below 20 nm reported and single lines as narrow as 7 nm
demonstrated [10].

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/n_re.png"  width="280" >
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.Y:</strong> HSQ simplified development
  </figcaption>
</figure>

When exposed to radiation, secondary electrons cleave the Si–H bonds within the cage structure, generating silanol groups that rapidly condense to form new Si–O–Si crosslinks. This converts the soluble cage structure into a dense, crosslinked network that is insoluble in developer solutions such as TMAH. The unexposed, uncrosslinked regions are dissolved during development and removed, leaving the crosslinked network as the patterned feature,the negative-tone
response [10] [13].

#### Resist choice

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/resist_choice.png"  width="280" >
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.Y:</strong> Simplifies diagram of resist liff off comparison
  </figcaption>
</figure>

PMMA is selected over HSQ for this project for lift-off compatibility. The fabrication process in this project requires metal deposition followed by resist lift-off to define the metallic grid features. As a positive resist, PMMA produces an undercut profile during development that allows clean separation of the deposited metal film [11] [12]. HSQ, as a negative resist, produces an overcut profile that prevents clean lift-off and is therefore incompatible with this process flow [13].




### 2.2 Proton Beam writing 
#### Dosage
In proton-beam writing, dose refers to the total charge delivered per unit area of resist, expressed in nC/mm². It is the product of the beam current, the dwell time per pixel, and the inverse of the pixel area. Physically, it represents the number of protons that have passed through each unit area of the resist surface, a higher dose means more protons, more secondary electron generation, and therefore more events per unit volume of photo resist.


`Dose is distinct from energy. Energy determines where in the resist the protons stop and how deeply they penetrate. Dose determines how much chemical damage is accumulated at each depth along that path.`





<figure style="text-align: center; margin: 20px 0;">
  <img src="images/dosage.png" alt="resolution fabrication overview" >
  
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.Y:</strong> dosage and development testing diagram
  </figcaption>
</figure>
 
There is a minium dose required, called the threshold dose to full develop a certain volume of PMMA, in this case roughly ~100 nC/mm² for 1 um of PMMA . Below this value, the chain scission density is insufficient for the developer (DI:IPA 7:3) to dissolve the exposed material, and the feature will either partially develop or not develop at all as seen below.

To test this and choose a suitable dose range, we fabricated various grids as seen above on the same piece of Si waver with the same seed metal 2nm of Cr




<figure style="text-align: center; margin: 20px 0;">
  <img src="images/dev.png" alt="resolution fabrication overview" >
  
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.Y:</strong> dosage and development testing results
  </figcaption>
</figure>


#### Effects of dosage

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/dosage_lay.png" alt="resolution fabrication overview" >
  
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.Y:</strong> Dosage effect diagram
  </figcaption>
</figure>


**Underdose (below ~50–75 nC/mm²):** Insufficient chain scission. The exposed PMMA
does not reach its threshold dose. The feature either does not develop, or develops incompletely, leaving a residual PMMA layer at the bottom. This prevents metal seedf layer from being exposed, produces no visible grid feature after lift-off.

**Correct dose (> 100 nC/mm²):** The exposed volume is cleanly dissolved by the developer from top to bottom, producing a well-defined trench with vertical sidewalls whose quality is limited by the lateral straggle of the beam, as characterised in Section 3.2.

**Overdose (above > 280 nC/mm²):** Excess secondary electron generation begins to expose resist beyond the intended beam boundary. The trench widens beyond the written pattern, reducing the effective CD and degrading sidewall verticality. 

**Extreme overdose (above ~3.5 × 10¹⁴ ions/cm²):** PMMA undergoes a positive-to-negative resist transition, and the exposed regions become insoluble rather than soluble.  This regime is not relevant to the present project but would fundamentally change the development polarity if accidentally reached.




### 2.3 Material deposition

Following lithographic patterning and development of the resist, metal is deposited onto the substrate to form the functional grid features. For this project, metal deposition is carried out using physical vapour deposition (PVD) ,a broad class of processes in which a solid source material is vaporised and the resulting vapour is transported and condensed onto the substrate as a thin film [13] [14].
 
<figure style="text-align: center; margin: 20px 0;">
  <img src="images/PVD_schmatic.png" alt="PVD schematic"  >
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.5:</strong> PVD schematic
  </figcaption>
</figure>


The general PVD process proceeds in four stages:
1. Energy is applied to the source material to vaporise it
2. The vaporised material is transported through a high-vacuum environment
3. The vapour impinges on the substrate surface
4. The material condenses and forms a thin film on the substrate

PVD is well-suited to this project for several reasons. High vacuum conditions minimise contamination from residual gases. The deposition rate can be controlled, providing influence over the film morphology, texture, and surface roughness, all of which affect the calibration utility of the finished standard [14].



The primary disadvantage of PVD is that it is a line-of-sight process, atoms travel in straight paths from the source to the substrate and cannot coat surfaces that are geometrically hidden from the source. For this project, this is not a limitation, as the grid structure is a relatively simple planar geometry with no hidden or re-entrant surfaces requiring coating.

Additionally, given that  Pure PMMA has a glass transition temperature (Tg) of approximately 105–107 °C but commercial grades can range from 85 to 165 °C[16], the deposition method should not damage the PMMA 
 
Three PVD techniques are considered: magnetron sputtering, E-beam deposition and Filtered cathodic vacuum arc (FCVA).

#### Magnetron sputtering

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/RF_sch.png" alt = "PVD schematic"  >
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.6</strong> RF sputtering schematic
  </figcaption>
</figure>
For this project, magnetron sputtering is the most readily accessible deposition
technique available in the lab. Its key advantage in the context of PMMA-patterned substrates is that it is a non-thermal process ,energy is delivered to the target by ion bombardment rather than by heat, so the substrate temperature remains comparatively low during deposition. This reduces the risk of the PMMA resist warping or deforming before lift-off. Magnetron sputtering is also highly versatile in terms of target material, provided the material is compatible with
the vacuum level achievable in the available system.

The primary disadvantage for lift-off applications is the diffuse transport of
sputtered atoms: which can prevent clean lift-off [15].

    
#### E-beam deposition  

In electron beam (e-beam) deposition, a high-voltage (6–40 kV) electron beam is focused onto a target material held in a water-cooled crucible. The kinetic energy of the electrons is converted to thermal energy on impact, causing the target to melt or sublimate and produce a vapour flux that condenses onto the substrate as a thin film. The beam is deflected through 180° or 270° by a magnetic field ,keeping the filament away from the deposition path to preserve film purity ,and can be scanned across the target in X and Y to distribute heating uniformly. The process must be carried out under high vacuum (>10⁻² mbar) to prevent energy loss from electron collisions with residual gas molecules. E-beam deposition can vaporise materials with melting points up to ~2,800 °C, making it suitable for refractory metals that cannot be processed by resistive thermal evaporation.


#### DLC deposition via FCVA

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/FCVA.png" alt = "FCVA schematic"  >
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.Y</strong> FCVA schematic
  </figcaption>
</figure> 


`ref <a href="https://www.researchgate.net/figure/Schematic-of-the-filtered-cathodic-vacuum-arc-FCVA-deposition-system_fig6_267805644"><img src="https://www.researchgate.net/profile/Mahnaz-Shafiei/publication/267805644/figure/fig6/AS:669391066759183@1536606675976/Schematic-of-the-filtered-cathodic-vacuum-arc-FCVA-deposition-system.png" alt="4. Schematic of the filtered cathodic vacuum arc (FCVA) deposition system."/></a>`


Filtered cathodic vacuum arc (FCVA) is a PVD technique in which a high-current arc is struck on a graphite cathode, generating a carbon plasma that is directed 
onto the substrate through a magnetic filter. The filter removes macroparticles from the plasma stream, producing a dense, smooth diamondlike carbon (DLC) film 
with a tunable sp²/sp³ ratio depending on the arc parameters. Unlike sputtering or thermal evaporation, FCVA can deposit hard, wear-resistant carbon films at room temperature without requiring a precursor gas.


### 2.4 Material choice

For this fabrication project, a metal is required that satisfies four criteria:
 
1. It must be compatible with PMMA lift-off ,i.e. evaporable at substrate

2. It must be a good electron scatterer for SEM and TEM characterization --   materials with high atomic number (high-Z) produce strong contrast in electron microscopy.

3. It must be chemically stable ,the grid standard must resist oxidation or corrosion during storage and repeated use.

4. It must have low lattice mismatch with the silicon substrate to minimize stress-induced deformation of the thin film.
The silicon substrate has a lattice parameter of a = 5.431 Å (diamond cubic structure) [18]. Lattice mismatch f is defined as f = (a_film − a_Si) / a_Si.

5. Good surface smoothness, limiting e- scattering 
Using the Névot-Croce factor, to visualize the importance of a smooth surface under <3nm

`ref <a href="https://www.researchgate.net/figure/Diffuse-surface-scattering-increases-as-roughness-increases-Richards-2009_fig6_349395724"><img src="https://www.researchgate.net/profile/Ayman-Abdel-Hamid-2/publication/349395724/figure/fig6/AS:992501460660234@1613642196704/Diffuse-surface-scattering-increases-as-roughness-increases-Richards-2009.jpg" alt="7. Diffuse surface scattering increases as roughness increases (Richards, 2009)."/></a>`


<figure style="text-align: center; margin: 20px 0;">
  <img src="images/sacttering_surface.jpg" alt = "FCVA schematic"  >
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 2.Y</strong> Surface scattering
  </figcaption>
</figure> 

The diagram above does a good job of visualizing the scattering. Electron microscope calibrations, rely on the backsactering of electron, scattering such as this can lead to skewed images. 

The Névot-Croce factor is a  correction factor that tells you how much of the specular (mirror-like) reflected signal you lose from a surface due to roughness.
A perfectly smooth surface reflects 100% of the incident beam in the specular direction. As the surface gets rougher, some of that signal scatters diffusely in random directions instead, so the sharp, coherent reflection you're trying to measure gets weaker


<iframe src ="scripts/nevot_croce_roughness.html"
        allowfullscreen="true" 
        width="500px" 
        height="500px">
</iframe>


The plot shows signal retention as a function of roughness across representative scattering conditions. At q_z = 0.5 nm⁻¹ (grazing incidence, where CD-AFM and SEM calibration measurements typically operate), a surface roughness below 1 nm yields a less than 6% of the coherent specular signal is lost to diffuse scatter. This places the standard firmly within the near-ideal regime where roughness-induced measurement bias is negligible.






### 2.5 method of analysis
Three complementary characterization techniques are used to evaluate the fabricated grid resolution standard: electron detector for edge straightness and sidewall angle, atomic force microscopy (AFM) for surface roughness.

#### Electron detector - edge straightness  and side wall angle 

Electron detector is the primary tool used in this project to assess edge quality and estimate sidewall angle. When the p-beam beam scans across the edge of a grid feature, the secondary electron yield increases sharply at the sidewall, producing a bright edge peak in the greyscale line profile. The width of this bright band, known as the edge width (EW) or white-band width (WBW), is directly related to the sidewall angle: a steeper, more vertical sidewall produces a narrower edge band, while a sloped or tapered sidewall broadens it.

The edge intensity profile is fitted using a combined error function and Gaussian model:
 
$$ F(x) = A\left[1 + \text{Erf}\!\left(\frac{2\sqrt{\ln 2}}{f}(d - x)\right)\right] + B\exp\!\left(-\frac{\ln 16}{f^2}(d - x)^2\right) + C $$
 
where *A* is the error function amplitude, *B* is the Gaussian amplitude, *C* is the baseline offset, *d* is the fitted edge position in pixels, and *f* is the FWHM of the edge transition [3].

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/ni_grid_x_lc.jpg" alt = "nevot_coroce"  >
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">  analysis method used on previously made nickel grid 
    <strong>Figure 2.7</strong> 
  </figcaption>
</figure>


The **error function term** models the underlying step transition in secondary electron intensity as the beam crosses the edge, the fundamental shape of an ideal edge profile convolved with the finite beam diameter. The **Gaussian term** accounts for the bright secondary electron emission peak at the sidewall. Together they give a physically complete description of the measured profile.
 
The key output is *f* , the FWHM of the fitted edge transition. A smaller *f* corresponds to a sharper, more abrupt edge, which in turn indicates a more vertical sidewall. The sidewall angle θ is estimated geometrically from the fitted FWHM and the known feature height *h*:
 
$$ \theta = 90° - \arctan\!\left(\frac{f}{h}\right)$$
 
where *h* is the feature height determined from the PBW process parameters and verified by AFM step-height measurement.

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/calc_angle.png" "  >
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;"> Vizual of angle calcualtions
    <strong>Figure 2.8</strong> 
  </figcaption>
</figure>


The resolution limit for this type of measurement is approximately 1 nm,below which the finite beam diameter and beam-sample interaction volume prevent reliable edge discrimination. It is also limited to top-down or oblique imaging, it cannot directly image the sidewall profile without cross-sectioning,
which is destructive. 

To utilize and analyse the electron detector spectrum, I developed a python script, flow chart seen below. 

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/software_flow.png" alt = "nevot_coroce"  >
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;"> software flowchart
    <strong>Figure 2.9</strong> 
  </figcaption>
</figure>

#### AFM - Surface Roughness

AFM is used to characterize the surface roughness of the top face of the deposited metal grid features and of the exposed silicon substrate between features. The AFM tip scans in tapping mode across the sample surface, recording sub-nanometer height variations. From the resulting height map, the root mean square roughness R_rms (also written R_q) is extracted, the standard deviation of height across the measured area. Below is an example of surface roughness as shown using Gwyddion

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/afm_example_pdsi.png" alt = "AFM example" >



  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;"> Afm example 
    <strong>Figure 2.10</strong> 
  </figcaption>
</figure>

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/afm_ex_graph.png" alt="afm graph" >

  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;"> graph of surface at center lines in both verticle and horizontal
    <strong>Figure 2.11</strong> 
  </figcaption>
</figure>


For a resolution standard, surface roughness is significant for two reasons. First, it affects the accuracy of AFM-based calibration measurements made using the standard: a rough reference surface introduces uncertainty into tip characterization. Second, roughness provides indirect information about the quality of the deposition process and the uniformity of the metal film grain structure. Target surface roughness for a usable resolution standard is typically below a few nanometers R_rms.


#### Surface roughness metrics

Surface roughness is quantified from the AFM height profiles using three standard parameters. For a profile of N height points y_i, with the mean height subtracted to remove any scan tilt or baseline offset, the three metrics are defined as follows.

The root mean square roughness R_q is the standard deviation of the height values:

$$ R_q = \sqrt{\frac{1}{N} \sum_{i=1}^{N} y_i^2} $$

The arithmetic mean roughness R_a is the average of the absolute height deviations:

$$ R_a = \frac{1}{N} \sum_{i=1}^{N} |y_i| $$

R_a treats all deviations equally regardless of their size and is less sensitive to outliers than R_q. It is reported here as a secondary reference, as it is the most widely cited roughness parameter in industrial standards.

The total height R_z is the peak-to-valley span across the full profile:

$$ R_z = y_{max} - y_{min} $$

R_z gives the worst-case surface excursion and is most sensitive to isolated spikes or scratches. A large R_z relative to R_q indicates the presence of a small number of extreme features on an otherwise smooth surface.

All three parameters are computed from 1D line profiles extracted from the AFM height map by the Python analysis script. Values are reported in nanometres after converting from the raw SI metre output of Gwyddion.


[<--Prev: Introduction ](Introduction.md) | [Next: Fabrication →](Fabrication.md)


### References

<div class="references">

<ol>
  <li>H. Duan, D. Winston, J. K. W. Yang, B. M. Cord, V. R. Manfrinato, and K. K. Berggren, "Sub-10-nm half-pitch electron-beam lithography by using poly(methyl methacrylate) as a negative resist," <em>Microelectronic Engineering</em>, 2015. DOI: 10.1016/j.mee.2015.02.042</li>

  <li>K. Yamazaki, "Electron beam direct writing," in <em>Nanofabrication: Fundamentals and Applications</em>, A. A. Tseng, Ed. Singapore: World Scientific, 2008, ch. 10.</li>

  <li>R. Winkler et al., "Roadmap for focused ion beam technologies," <em>Applied Physics Reviews</em>, vol. 10, no. 4, art. 041311, 2023. DOI: <a href="https://doi.org/10.1063/5.0162597">10.1063/5.0162597</a></li>

  <li>J. Gierak et al., "Effects of focused gallium ion-beam implantation on properties of nanochannels on silicon-on-insulator substrates," <em>Applied Physics Letters</em>, vol. 89, 2006. Available: <a href="https://www.researchgate.net/publication/249512973">researchgate.net</a></li>

  <li>J. A. van Kan, A. A. Bettiol, and F. Watt, "Proton beam writing of three-dimensional nanostructures in hydrogen silsesquioxane," <em>Nano Letters</em>, vol. 6, no. 3, pp. 579–582, 2006. DOI: 10.1021/nl052478c</li>

  <li>F. Watt, A. A. Bettiol, J. A. van Kan, E. J. Teo, and M. B. H. Breese, "Ion beam lithography and nanofabrication: a review," <em>International Journal of Nanoscience</em>, vol. 4, no. 3, pp. 269–286, 2005.</li>

  <li>F. Watt, M. B. H. Breese, A. A. Bettiol, and J. A. van Kan, "Proton beam writing," <em>Materials Today</em>, vol. 10, no. 6, pp. 20–29, 2007. DOI: 10.1016/S1369-7021(07)70129-3</li>

  <li>J. A. van Kan, P. G. Shao, Y. H. Wang, and P. Malar, "Proton beam writing: a platform technology for high quality three-dimensional metal mold fabrication for nanofluidic applications," <em>Microsystem Technologies</em>, vol. 17, pp. 1519–1527, 2011. DOI: 10.1007/s00542-011-1333-0</li>

  <li>A. A. Bettiol, S. Venugopal Rao, E. J. Teo, J. A. van Kan, and F. Watt, "Sidewall quality in proton beam writing," <em>Nuclear Instruments and Methods in Physics Research Section B</em>, vol. 258, no. 1, pp. 302–306, 2007. DOI: 10.1016/j.nimb.2007.01.073</li>

  <li>S. Rajendran, J. A. van Kan, T. Osipowicz, and F. Watt, "Design considerations for a compact proton beam writing system aiming for fast sub-10 nm direct write lithography," <em>Nuclear Instruments and Methods in Physics Research Section B</em>, 2016. DOI: 10.1016/j.nimb.2016.11.022</li>

  <li>C. Mack, <em>Fundamental Principles of Optical Lithography</em>. Chichester: Wiley, 2007.</li>

  <li>Microchem / Kayaku Advanced Materials, "PMMA Data Sheet," 2019. Available: <a href="https://kayakuam.com/wp-content/uploads/2019/09/PMMA_Data_Sheet.pdf">kayakuam.com</a></li>

  <li>J. A. van Kan, P. Malar, and A. B. H. Tay, "Resist materials for proton beam writing: a review," <em>Applied Surface Science</em>, 2014. DOI: 10.1016/j.apsusc.2014.04.147</li>

  <li>A. A. Tseng, K. Chen, C. D. Chen, and K. J. Ma, "Electron beam lithography in nanoscale fabrication: recent development," <em>IEEE Transactions on Electronics Packaging Manufacturing</em>, vol. 26, no. 2, pp. 141–149, 2003. DOI: 10.1109/TEPM.2003.817714</li>

  <li>L. B. Freund and S. Suresh, <em>Thin Film Materials: Stress, Defect Formation and Surface Evolution</em>. Cambridge: Cambridge University Press, 2003.</li>

  <li>H. Frey and H. R. Khan, Eds., <em>Handbook of Thin-Film Technology</em>. Berlin: Springer, 2015.</li>

  <li>K. Müller, "Thermal stability of PMMA resists used in nanolithography," <em>Scientific Reports</em>, 2021. DOI: 10.1038/s41598-021-01282-7</li>

  <li>National Institute of Standards and Technology, "Improving CD-AFM measurements from the tip down," NIST News, Mar. 2016. Available: <a href="https://www.nist.gov/news-events/news/2016/03/improving-cd-afm-measurements-tip-down">nist.gov</a></li>
</ol>

</div>
 