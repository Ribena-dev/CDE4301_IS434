## Future works
### 5.1 things to improve base don final results
- isk man i don't have the results yet 

### 5.2 scalability
Nanoimprint lithography (NIL) offers a path to high-throughput replication of the 
grid resolution standard without requiring repeated PBW exposures. In this approach, 
a PBW-fabricated PMMA master is used as a stamp to imprint the grid pattern into a 
fresh polymer substrate, transferring the geometry in a single press cycle.

Initial trials were conducted using an Omostamp silicon stamp on a NILT CNI 
nanoimprinter (software v1.0.0.42). The hot embossing recipe used is summarised 
in Table 5.1.

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/stamp.png" alt="nano imprinting overview" width="280" style="margin: 5px;">

  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 5.1:</strong> nano imprinting overview
  </figcaption>
</figure>

**Table 5.1 — Nanoimprint recipe parameters (NILT CNI, hot embossing)**

| Step | Parameter | Value |
|---|---|---|
| 1 — Vacuum | Threshold | 90.0 |
| | ForceVac | 0 |
| | VacTime | 0.2 min |
| 2 — Temperature ramp | Target temperature | 130 °C |
| | Ramp time | 0.0 min |
| | Pressure | 1.0 bar |
| | Wait for temp | Yes |
| 3 — Hold | Hold time | 10.0 min |
| | Imprint pressure | 9.0 bar |
| 4 — Cooldown ramp | Target temperature | 60 °C |
| | Pressure | 9.0 bar |
| | Wait for temp | Yes |

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/imprinter.jpg" alt="" width="280" style="margin: 5px;">

  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 5.2:</strong> nano imprinting mahcine
  </figcaption>
</figure>


Two problems were identified in the initial trials. First, the PMMA resist layer 
delaminated from the silicon wafer during demolding — the grid features were 
pulled off the substrate rather than transferring cleanly into the imprint polymer. 
This is likely caused by insufficient adhesion between the PMMA and the silicon 
surface, either due to inadequate surface preparation or wafer contamination prior 
to spin coating. Depositing a thin adhesion layer (e.g. HMDS or Cr) between the 
silicon and PMMA prior to PBW is a recommended corrective step.

Second, the PMMA film thickness exceeded the Omostamp feature height, causing 
overflow of resist material beyond the patterned region during the imprint step. 
Future trials should match the PMMA spin thickness more closely to the stamp 
feature height — either by reducing the spin-coated thickness or by selecting a 
lower-concentration PMMA grade.

Resolving these two issues would make NIL a viable route for producing multiple 
copies of the grid standard from a single PBW master, significantly reducing the 
cost and time per calibration artefact.

[<--Prev: Results and analysis ](fna.md) | 
[Next: Appendix →](fna.md)


