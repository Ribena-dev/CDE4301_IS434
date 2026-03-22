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

Earlier project stages relied heavily on nickel electroplating for producing the grid structures [12]. When I learned that the nickel plating machine was unavailable, it forced a reconsideration of the metallisation strategy. I viewed this as an opportunity to explore combinations of metal and non metal layering, which may help imporve the electron count contrast.

Gold emerged as the most feasible base material because it integrates smoothly with existing sputtering workflows and offers reliable conductivity. Additionally, I wanted to test diamond-like carbon (DLC) coating on top of the gold. DLC is an insulating material and has a smooth surface. In theory, DLC could smoothen the surface and enhance contrast while still maintaining compatibility with the existing fabrication sequence.

### 3.3 Beam Writing Strategy Investigation

As seen later, an unexpected curvature appeared along several of the fabricated structures. This observation prompted me to question whether the writing strategy, specifically the balance between beam intensity and the number of passes, was contributing to these defects. In principle, multiple low-intensity passes could yield a more averaged and uniform exposure, assuming that beam alignment remains stable and free of drift. Conversely, if the step size is sufficiently small, a single high-intensity pass should theoretically deliver comparable uniformity, though the concentrated energy deposition may introduce localised distortions.

<figure style="text-align: center; margin: 20px 0;">

<img src="images/passes.png" alt = "varying number of passes" style = "width:300px;" />
<figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 3.3:</strong> diagram displaying varying passes
</figcaption>
</figure>
A useful analogy is laser cutting: a single high-power pass cuts quickly without alignment issues but may cause edge damage from concentrated energy, while multiple low-power passes can produce cleaner cuts by distributing energy more evenly,if the laser stays aligned. Similarly, single-pass high-intensity beam writing avoids drift but may cause the observed curved defects, whereas multiple low-intensity passes could yield straighter edges through better dose averaging, assuming stable beam alignment.


### 3.4 Focal Plane Variation

Even with proton beams, some beam divergence exists, meaning beam spreading still occurs in resist material, although still less than an electron beam. Theoretically, converging beam angle could counteract diverging beam characteristics within resist material, reducing the spread even further and achieving a straighter edge. Systematically varying focal plane position as function of depth during writing could keep beam optimally focused throughout resist thickness, potentially achieving better sidewall verticality and edge definition.

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/beam_focal.png" alt="Beam convergence and focal plane variation" width="400">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 3.3:</strong> focal plane positioning strategy
  </figcaption>
</figure>


[← Research objectives](objectives.md) | [Next: Fabrication steps →](fabrication.md)