## 3. Materials Selection and Considerations

### 3.1 PMMA Resist Selection

Given ongoing project history, fabrication parameters derive from previous research [11]. PMMA serves as positive resist because proton beam exposure causes chain breakage, reducing polymer molecular weight and enabling selective removal of exposed regions. PMMA height determines achievable structure depth and final metal deposition height. Maximum height before structural integrity compromises and walls warp requires careful attention to ensure geometry maintenance throughout processing.

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/spin_curve.png" alt="PMMA spin coating curve" width="400">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 3.1:</strong> PMMA resist thickness vs spin coating speed
  </figcaption>
</figure>

### 3.2 Metal Selection and Challenges

Previous research used nickel for grid structures [12], but nickel plating machine unavailability necessitated exploring alternative metals. Different metals produce varying surface smoothness and backscattered electron contrast. This work tests diamond-like carbon (DLC) coating on gold. Gold offers excellent conductivity and established sputtering protocols, but surface characteristics present challenges motivating DLC coating exploration for improved smoothness and imaging contrast.

### 3.3 Beam Writing Strategy Investigation

Critical parameters include whether high beam intensity over single pass produces better results than low intensity over multiple passes. Low intensity with multiple passes could produce more averaged, even exposure if beam alignment maintains with no drift. However, if step size is sufficiently small, high intensity single pass theoretically produces similar uniform exposure. Observed curved structural defects in initial samples may relate to writing strategy, making this investigation particularly important for achieving straight vertical edges.

<figure style="text-align: center; margin: 20px 0;">

<img src="images/passes.png" alt = "varying number of passes" style = "width:300px;" />
<figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 3.3:</strong> diagram displaying varying passes
</figcaption>
</figure>
A useful analogy is laser cutting: a single high-power pass cuts quickly without alignment issues but may cause edge damage from concentrated energy, while multiple low-power passes can produce cleaner cuts by distributing energy more evenly—if the laser stays aligned. Similarly, single-pass high-intensity beam writing avoids drift but may cause the observed curved defects, whereas multiple low-intensity passes could yield straighter edges through better dose averaging, assuming stable beam alignment.


### 3.4 Focal Plane Variation

Even with proton beams, some beam divergence exists, meaning beam spreading still occurs in resist material, although still less than an electron beam. Theoretically, converging beam angle could counteract diverging beam characteristics within resist material, reducing the spread even further and achieving a straighter edge. Systematically varying focal plane position as function of depth during writing could keep beam optimally focused throughout resist thickness, potentially achieving better sidewall verticality and edge definition.

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/beam_focal.png" alt="Beam convergence and focal plane variation" width="400">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 3.3:</strong> focal plane positioning strategy
  </figcaption>
</figure>


[← Research objectives](objectives.md) | [Next: Fabrication steps →](fabrication.md)