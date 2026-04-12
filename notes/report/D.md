## Apendix : sidewall widget math 

The widget is built on one core geometric relationship between sidewall angle,
feature height, and the CD error introduced into a top-down SEM measuremen

### The geometry

When a feature has a perfectly vertical sidewall (θ = 90°), the top width and the bottom width are identical. When the sidewall tilts by an angle δ away from vertical (so θ = 90° − δ), the base of the feature is wider than the top.

W_top
    ┌───────────┐   ← SEM scans here, reads W_top
   /             \
  /               \
 /                 \
└───────────────────┘  ← true CD at base = W_bot
        W_bot



The overhang on each side is the horizontal distance gained by travelling down
the full feature height h at angle δ from vertical:

$$\text{overhang} = h \cdot \tan\delta = h \cdot \tan(90° - \theta)$$

Since there are two sides, the total extra width at the base is:

$$W_{bot} - W_{top} = 2h\tan(90° - \theta)$$

### The CD error

A top-down SEM cannot see the base of the feature — it measures the top width
W_top only. If the true critical dimension of interest is W_bot, the CD error is:

$$\text{CD error} = W_{bot} - W_{top} = 2h\tan(90° - \theta)$$

In the interactive model, h = 40 nm is fixed. At θ = 85° (δ = 5°):

$$\text{CD error} = 2 \times 40 \times \tan(5°) = 80 \times 0.0875 \approx 7.0 \text{ nm}$$

As a percentage of a 20 nm CD feature:

$$\text{CD error (%)} = \frac{2h\tan(90°-\theta)}{CD} \times 100
= \frac{7.0}{20} \times 100 = 35\%$$

### The simulated SEM intensity profile

The profile in the right panel models what a secondary electron detector sees
as the beam scans across the feature. Three contributions are summed:

$$I(x) = I_{\text{bulk}}(x) + I_{\text{top edges}}(x) + I_{\text{base edges}}(x)$$

**$I_{\text{bulk}}$** is a flat elevated signal over the feature. Metal produces
more backscattered electrons than the silicon substrate, so the feature appears
brighter overall.

**$I_{\text{top edges}}$** is a Gaussian spike at each top corner, arising from
the edge enhancement effect — a sharp corner produces a burst of secondary
electrons as the beam passes over it:

$$I_{\text{edge}}(x) = A \cdot \exp\!\left(-\frac{(x - x_{\text{edge}})^2}{2\sigma^2}\right)$$

**$I_{\text{base edges}}$** is a smaller Gaussian at the projected base corners,
which becomes visible when the sidewall tilts significantly because the angled
wall intercepts part of the incident beam.

### Why the top markers are fixed

The red dashed lines marking W_top do not move regardless of sidewall angle.
This is the key point — a standard top-down SEM always reports the top width,
so it always reads 20 nm no matter how sloped the walls are. It is the green
W_bot markers that slide outward as the angle decreases, showing the true base
width growing while the SEM measurement stays fixed. The gap between the two
is the CD error.


[← Prev: Appendix C](C.md) | [Next: Appendix E →](E.md)