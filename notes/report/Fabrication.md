## Fabrication
### 3.1 overview of fabrication steps 

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/lit_fab_3.png" alt="resolution fabrication overview"  style="margin: 5px;">
  
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 3.1:</strong> Fabrication process for resolution standard overview
  </figcaption>
</figure>
 


### 3.2 simulations of P-beam in PMMA 

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/srim_trajectories.png" 
       alt="SRIM simulation showing 20 sample 2 MeV proton trajectories through 1 µm PMMA in the X-Y plane (left) and the lateral exit spread distribution in the Y-Z plane with outlier nuclear scatter events marked (right)" 
        style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 3.2:</strong> SRIM Monte Carlo simulation of 2 MeV proton trajectories 
    in 1 µm PMMA. Left: side-view (X–Y) showing 20 sample ion paths,protons travel 
    near-straight with minimal lateral deviation. Right: exit spread (Y–Z) showing the 
    radial distribution at the resist base;.
  </figcaption>
</figure>

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/srim_lateral_straggle.png" 
       alt="Plot of lateral straggle sigma versus depth in PMMA for 2 MeV protons, comparing raw SRIM data including nuclear scatter outliers against IQR-cleaned data, with a 3 nm target threshold line" 
        style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 3.3:</strong> Lateral straggle σ<sub>r</sub> as a function of depth 
    for 2 MeV protons in PMMA. The grey dashed curve shows the raw SRIM data 
  </figcaption>
</figure>

Spikes are caused by rare large-angle nuclear scattering events. The teal curve shows the cleaned data after IQR ×3 outlier removal, revealing a true straggle of 0.81 nm at the 1 µm exit depth, well below the 3 nm target (red dotted line).
 
SRIM Monte Carlo simulations were used to characterise the behaviour of 2 MeV protons in PMMA and to predict the theoretical sidewall angle of the fabricated features. Two outputs are of interest: the depth distribution (Bragg peak), which confirms the feature height
achievable at a given beam energy; and the lateral straggle σ(z), which governs edge sharpness as a function of depth.
 
#### Theoretical Sidewall Angle
 
The lateral straggle σ(z) from SRIM gives the standard deviation of the beam's lateral spread at depth z. The edge transition width at that depth is related to σ by:
 
$$ f(z) = 2\sqrt{2\ln 2} \cdot \sigma(z) \approx 2.355\,\sigma(z) $$
 
where f is the FWHM of the dose profile across the feature edge ,the same parameter extracted from SEM measurements in Section 2.5.1. The theoretical sidewall angle at the full feature depth h is then:
 
$$ \theta = 90° - \arctan\!\left(\frac{f(h)}{h}\right) = 90° - \arctan\!\left(\frac{2.355\,\sigma(h)}{h}\right) $$


### 3.3 spin coating the waver and development
#### spin coating

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/pmma_495k_mid.png" 
       alt="Spin speed vs film thickness for PMMA 495K at medium concentration in anisole" 
       width="280" style="margin: 5px;">
  <img src="images/pmma_495k_thin.png" 
       alt="Spin speed vs film thickness for PMMA 495K at low concentration in anisole" 
       width="280" style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 3.X:</strong> Spin curves for PMMA 495K at medium (left) 
    and low (right) concentrations in anisole. Higher spin speed produces thinner films.
  </figcaption>
</figure>

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/pmma950k.png" 
       alt="Spin speed vs film thickness for PMMA 950K in anisole" 
       width="280" style="margin: 5px;">
  <img src="images/pmma950k_thinrange.png" 
       alt="Spin speed vs film thickness for PMMA 950K at low concentration showing thin film range" 
       width="280" style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 3.Y:</strong> Spin curves for PMMA 950K at standard (left) 
    and dilute (right) concentrations. The higher molecular weight produces 
    thicker films at equivalent spin speed compared to 495K.
  </figcaption>
</figure>


Film thickness is governed by two parameters: the concentration (viscosity) of the resist solution and the spin speed [1]. Higher spin speeds and lower concentrations produce thinner films, following an approximate inverse power-law relationship.
 
PMMA is available in two standard molecular weights 495K and 950K, each supplied at multiple concentrations in anisole (e.g. A2, A4, A6 for 2%, 4%, 6% solids by weight) [1][2]. Higher molecular weight resist is more viscous at the same concentration and produces a slightly thicker film at a given spin speed. The choice of molecular weight and concentration together determine the accessible thickness

A key design constraint is that the PMMA thickness must be at least five times greater than the intended metal deposition thickness. This ratio is required for two reasons: first, it ensures sufficient structural integrity of the resist walls during development and metal
deposition; and second, it prevents metal overflowing the resist sidewalls, a phenomenon known as mushrooming, where excess metal forms a cap-like layer over the resist that prevents clean lift-off. However, if the PMMA is made excessively thick, the increased aspect ratio of the trench can cause resist wall collapse, and as shown by the SRIM simulations in Section 3.2, deeper features allow more lateral beam straggle to accumulate, degrading the
sidewall angle. The optimal PMMA thickness is therefore determined by balancing these competing constraints against the spin curves shown above.


#### Pre-back , Post-bake

After spin coating, the wafer is placed on a hotplate for a soft bake, typically at 180 °C for 60–90 seconds [1] [3]. The pre-bake serves two purposes: it drives off residual solvent (anisole) from the film, which would otherwise cause the resist to remain tacky and deform during handling; and it densifies and hardens the film, improving adhesion to the substrate and reducing unwanted swelling during development. Baking above ~125 °C is avoided as PMMA begins to flow and reflow at elevated temperatures, rounding the resist edges [1].

#### Development and lift off



<video width="300px" controls>
  <source src="images/development_1.mp4" type="video/mp4">
</video>




Development is performed after PBW exposure and is included here for process continuity. The wafer is immersed in DI water:IPA (7:3) developer, which selectively dissolves the chain-scissioned PMMA in the exposed regions while leaving the unexposed resist intact [1][2]. The sample is then rinsed in fresh IPA and dried with a nitrogen gun to stop development. Following metal deposition, the remaining PMMA is removed by immersion in acetone, lifting off the metal on top of the resist and leaving only the metal deposited directly onto the silicon substrate.

### 3.4 P-beam structure

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/beam_line.png" alt="beam line optics"  style="margin: 5px;">
  
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 3.4:</strong> Beam line optics
  </figcaption>
</figure>


The PBW facility at CIBA is built around a 3.5 MV Singletron accelerator (HVEE) which generates a focused MeV proton beam for lithography [4] [5]. Protons are produced from hydrogen gas, accelerated to the required energy, and filtered by a 90° analysing magnet before being directed to the PBW end station via a switching magnet. Blanking plates deflect the beam off-axis to control dose delivery during patterning [4].
 
Before focusing, the beam passes through two apertures. The objective aperture (8 × 4 µm²) defines the virtual source size, while the collimator aperture (30 × 30 µm²) reduces angular divergence entering the lenses, giving a beam half-divergence of approximately 3 µrad [4].

Focusing is achieved by a spaced Oxford triplet of magnetic quadrupole lenses in a converging-diverging-converging (CDC) configuration. A single quadrupole focuses in one plane and defocuses in the other; the triplet arrangement produces a symmetric spot focus. With an object-to-lens distance of 7.5 m and image distance of 30 mm, the system achieves a demagnification of 857 × 130, yielding a minimum spot size of 9.3 × 32 nm² [4]. Chromatic aberration, is the dominant limit on spot size, requiring ~10 ppm accelerator stability for sub-10 nm resolution [4]. Before writing, the beam is focused by scanning across a free-standing resolution standard. The transmitted or secondary electron signal produces a complementary error function profile, which is fitted to extract the beam FWHM (Section 2.5). Once focused, a writing file is loaded and the beam is rastered over the resist using electrostatic scanners combined with stage movement for
larger fields [5].

<!-- <iframe 
  src="https://github.com/Ribena-dev/CDE4301_IS434/blob/main/notes/report/scripts/beam_geo.html" 
  width="100%" 
  height="580px" 
  style="border:none; border-radius:6px;"
  sandbox="allow-scripts" >
</iframe> -->

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PBW Beam Geometry Explorer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: sans-serif; font-size: 13px; background: #f8f8f6; color: #222; padding: 12px; }
  h2 { font-size: 14px; font-weight: 600; margin-bottom: 10px; color: #333; }

  .layout { display: grid; grid-template-columns: 1fr 1fr 260px; gap: 10px; }
  .side-col { grid-column: 1 / 3; }

  canvas { display: block; width: 100%; background: #f8f8f6; }

  .sliders { display: flex; flex-direction: column; gap: 6px; margin-top: 10px; }
  .slider-row { display: flex; align-items: center; gap: 8px; }
  .slider-row label { width: 130px; font-size: 12px; color: #555; flex-shrink: 0; }
  .slider-row input[type=range] { flex: 1; }
  .slider-row .val { width: 54px; font-size: 12px; font-weight: 600; text-align: right; }

  .mathbox { background: white; border: 1px solid #ddd; border-radius: 6px;
             padding: 10px; font-family: monospace; font-size: 11px;
             line-height: 1.65; white-space: pre; color: #222;
             grid-row: 1 / 3; overflow-y: auto; }

  .panel-label { font-size: 11px; font-weight: 600; color: #888; margin-bottom: 4px; }
</style>
</head>
<body>

<h2>PBW beam geometry ,drag sliders to explore slit opening and focal position</h2>

<div class="layout">

  <!-- Side view canvas -->
  <div class="side-col">
    <div class="panel-label">Side view ,cone geometry</div>
    <canvas id="cvSide" width="700" height="260"></canvas>
  </div>

  <!-- Math box -->
  <div class="mathbox" id="mathBox"></div>

  <!-- Front view canvas -->
  <div>
    <div class="panel-label">Beam cross-section at sample</div>
    <canvas id="cvFront" width="260" height="260"></canvas>
  </div>

</div>

<!-- Sliders -->
<div class="sliders">
  <div class="slider-row">
    <label style="color:#BA7517">&#9632; Sample Δz (µm)</label>
    <input type="range" id="slDz" min="-20" max="20" step="0.1" value="-8">
    <span class="val" id="vlDz" style="color:#BA7517">-8.0</span>
  </div>
  <div class="slider-row">
    <label style="color:#185FA5">&#9632; X slit (µm)</label>
    <input type="range" id="slSx" min="10" max="300" step="1" value="100">
    <span class="val" id="vlSx" style="color:#185FA5">100</span>
  </div>
  <div class="slider-row">
    <label style="color:#0F6E56">&#9632; Y slit (µm)</label>
    <input type="range" id="slSy" min="10" max="300" step="1" value="100">
    <span class="val" id="vlSy" style="color:#0F6E56">100</span>
  </div>
</div>

<script>
// ── Physics constants ─────────────────────────────────────────────────────
const D0X_REF = 9.3e-9, D0Y_REF = 32e-9;
const AX_REF  = 3e-6 * 857, AY_REF = 3e-6 * 130;
const SX_REF = 100, SY_REF = 100;
const EMIT_X = D0X_REF * AX_REF, EMIT_Y = D0Y_REF * AY_REF;

const B='#185FA5', T='#0F6E56', A='#BA7517', R='#A32D2D', GR='#888780';

function beamParams(sx, sy) {
  const ax = AX_REF * (sx / SX_REF);
  const ay = AY_REF * (sy / SY_REF);
  const d0x = EMIT_X / ax, d0y = EMIT_Y / ay;
  const dofx = d0x / (2*ax), dofy = d0y / (2*ay);
  return {ax, ay, d0x, d0y, dofx, dofy};
}

function spotSize(dz_um, p) {
  const dz = dz_um * 1e-6;
  const cbx = 2*p.ax*Math.abs(dz)*1e9;
  const cby = 2*p.ay*Math.abs(dz)*1e9;
  const dx  = Math.sqrt(p.d0x**2 + (2*p.ax*Math.abs(dz))**2)*1e9;
  const dy  = Math.sqrt(p.d0y**2 + (2*p.ay*Math.abs(dz))**2)*1e9;
  return {cbx, cby, dx, dy};
}

// ── Canvas helpers ────────────────────────────────────────────────────────
function arrow(ctx, x1,y1,x2,y2,color,lw=1.2) {
  ctx.save();
  ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=lw;
  const dx=x2-x1,dy=y2-y1,len=Math.hypot(dx,dy);
  const ang=Math.atan2(dy,dx), hs=7;
  ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x2,y2);
  ctx.lineTo(x2-hs*Math.cos(ang-0.4),y2-hs*Math.sin(ang-0.4));
  ctx.lineTo(x2-hs*Math.cos(ang+0.4),y2-hs*Math.sin(ang+0.4));
  ctx.closePath(); ctx.fill();
  ctx.restore();
}

// ── Side view ─────────────────────────────────────────────────────────────
function drawSide(dz_um, p) {
  const cv = document.getElementById('cvSide');
  const ctx = cv.getContext('2d');
  const W=cv.width, H=cv.height;
  ctx.clearRect(0,0,W,H);

  // coordinate mapping: z range [-21,5] µm → x pixels; y range [-20,20] scaled → y pixels
  const zMin=-21, zMax=5, yRange=40;
  const SCALE=0.6; // nm → display
  const zToX = z => (z - zMin) / (zMax - zMin) * W;
  const yToY = y => H/2 - y/yRange*H;

  // cone walls
  const nPts=400;
  const zArr=[], hwxArr=[], hwyArr=[];
  for(let i=0;i<nPts;i++){
    const z = zMin + i/(nPts-1)*(zMax-zMin);
    zArr.push(z);
    hwxArr.push(Math.abs(p.ax*z*1e-6)*1e9*SCALE);
    hwyArr.push(Math.abs(p.ay*z*1e-6)*1e9*SCALE);
  }

  // fill X cone
  ctx.save(); ctx.globalAlpha=0.10; ctx.fillStyle=B;
  ctx.beginPath(); ctx.moveTo(zToX(zArr[0]), yToY(hwxArr[0]));
  for(let i=1;i<nPts;i++) ctx.lineTo(zToX(zArr[i]), yToY(hwxArr[i]));
  for(let i=nPts-1;i>=0;i--) ctx.lineTo(zToX(zArr[i]), yToY(-hwxArr[i]));
  ctx.closePath(); ctx.fill(); ctx.restore();

  // X cone lines
  ctx.strokeStyle=B; ctx.lineWidth=1.8;
  ctx.beginPath(); ctx.moveTo(zToX(zArr[0]),yToY(hwxArr[0]));
  for(let i=1;i<nPts;i++) ctx.lineTo(zToX(zArr[i]),yToY(hwxArr[i])); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(zToX(zArr[0]),yToY(-hwxArr[0]));
  for(let i=1;i<nPts;i++) ctx.lineTo(zToX(zArr[i]),yToY(-hwxArr[i])); ctx.stroke();

  // fill Y cone
  ctx.save(); ctx.globalAlpha=0.14; ctx.fillStyle=T;
  ctx.beginPath(); ctx.moveTo(zToX(zArr[0]),yToY(hwyArr[0]));
  for(let i=1;i<nPts;i++) ctx.lineTo(zToX(zArr[i]),yToY(hwyArr[i]));
  for(let i=nPts-1;i>=0;i--) ctx.lineTo(zToX(zArr[i]),yToY(-hwyArr[i]));
  ctx.closePath(); ctx.fill(); ctx.restore();

  // Y cone lines
  ctx.strokeStyle=T; ctx.lineWidth=1.2;
  ctx.beginPath(); ctx.moveTo(zToX(zArr[0]),yToY(hwyArr[0]));
  for(let i=1;i<nPts;i++) ctx.lineTo(zToX(zArr[i]),yToY(hwyArr[i])); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(zToX(zArr[0]),yToY(-hwyArr[0]));
  for(let i=1;i<nPts;i++) ctx.lineTo(zToX(zArr[i]),yToY(-hwyArr[i])); ctx.stroke();

  // Optical axis
  ctx.strokeStyle='#bbb'; ctx.lineWidth=0.7; ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(0,H/2); ctx.lineTo(W,H/2); ctx.stroke();
  ctx.setLineDash([]);

  // DoF band
  const dofXpx1=zToX(-p.dofx*1e6), dofXpx2=zToX(p.dofx*1e6);
  ctx.save(); ctx.globalAlpha=0.06; ctx.fillStyle=B;
  ctx.fillRect(dofXpx1,0,dofXpx2-dofXpx1,H); ctx.restore();
  ctx.font='10px sans-serif'; ctx.fillStyle=B; ctx.textAlign='center';
  ctx.fillText(`DoF_x ±${(p.dofx*1e6).toFixed(2)}µm`, (dofXpx1+dofXpx2)/2, H-6);

  // Focus marker
  const fx=zToX(0), fy=yToY(0);
  ctx.fillStyle=R; ctx.beginPath(); ctx.arc(fx,fy,5,0,2*Math.PI); ctx.fill();
  ctx.fillStyle=R; ctx.font='11px sans-serif'; ctx.textAlign='left';
  ctx.fillText('z=0 (focus)', fx+6, fy-5);

  // Sample plane
  const sx = dz_um;
  const sxPx = zToX(sx);
  const sp = spotSize(dz_um, p);
  const ok = Math.abs(dz_um) <= p.dofx*1e6;
  const col = ok ? '#2ecc71' : R;
  ctx.strokeStyle=col; ctx.lineWidth=2.8;
  ctx.beginPath(); ctx.moveTo(sxPx,0); ctx.lineTo(sxPx,H); ctx.stroke();

  // sample beam footprint bar
  if(Math.abs(dz_um)>0.1){
    const hw=Math.abs(p.ax*dz_um*1e-6)*1e9*SCALE+p.d0x*1e9*SCALE*0.1;
    ctx.strokeStyle=R; ctx.lineWidth=5; ctx.lineCap='round'; ctx.globalAlpha=0.45;
    ctx.beginPath(); ctx.moveTo(sxPx,yToY(hw)); ctx.lineTo(sxPx,yToY(-hw)); ctx.stroke();
    ctx.globalAlpha=1; ctx.lineCap='butt';
  }

  // angle arrows + arc
  if(Math.abs(dz_um)>0.5){
    const hw=Math.abs(p.ax*dz_um*1e-6)*1e9*SCALE;
    arrow(ctx, fx, fy, sxPx, yToY(hw), B);
    arrow(ctx, fx, fy, sxPx, yToY(-hw), B);

    // arc
    const angMax = Math.atan2(hw, Math.abs(yToY(0)-yToY(0)+1) || 1);
    const rArc=28;
    ctx.strokeStyle=B; ctx.lineWidth=1.5;
    ctx.beginPath();
    if(sx<0){
      const ang0=Math.PI-Math.atan2(yToY(0)-yToY(hw), sxPx-fx);
      ctx.arc(fx,fy,rArc,Math.PI,Math.PI-Math.atan2(hw,Math.abs(sxPx-fx)),true);
    } else {
      ctx.arc(fx,fy,rArc,-Math.atan2(hw,Math.abs(sxPx-fx)),Math.atan2(hw,Math.abs(sxPx-fx)));
    }
    ctx.stroke();

    ctx.font='10px sans-serif'; ctx.fillStyle=B; ctx.textAlign = sx<0?'right':'left';
    const lx = sx<0 ? fx-rArc-4 : fx+rArc+4;
    ctx.fillText(`α_x ${(p.ax*1e3).toFixed(2)} mrad`, lx, fy+4);

    // cone blur annotation
    arrow(ctx, sxPx, yToY(0), sxPx, yToY(hw), A);
    ctx.font='10px sans-serif'; ctx.fillStyle=A; ctx.textAlign='left';
    ctx.fillText(`α·|Δz| = ${(sp.cbx/2).toFixed(1)}nm`, sxPx+5, (yToY(0)+yToY(hw))/2);
  }

  // Δz arrow
  if(Math.abs(dz_um)>0.5){
    arrow(ctx, fx, H-20, sxPx, H-20, col);
    ctx.font='10px sans-serif'; ctx.fillStyle=col; ctx.textAlign='center';
    ctx.fillText(`|Δz| = ${Math.abs(dz_um).toFixed(1)} µm`, (fx+sxPx)/2, H-4);
  }

  // Legend
  ctx.font='11px sans-serif'; ctx.textAlign='left';
  ctx.fillStyle=B; ctx.fillRect(W-155,8,10,10); ctx.fillText(`X cone  α_x=${(p.ax*1e3).toFixed(3)} mrad`, W-142, 18);
  ctx.fillStyle=T; ctx.fillRect(W-155,24,10,10); ctx.fillText(`Y cone  α_y=${(p.ay*1e3).toFixed(3)} mrad`, W-142, 34);
  ctx.fillStyle=col; ctx.fillRect(W-155,40,10,10); ctx.fillText(`sample Δz=${dz_um.toFixed(1)} µm`, W-142, 50);
}

// ── Front view ────────────────────────────────────────────────────────────
function drawFront(dz_um, p) {
  const cv = document.getElementById('cvFront');
  const ctx = cv.getContext('2d');
  const W=cv.width, H=cv.height;
  ctx.clearRect(0,0,W,H);

  const sp = spotSize(dz_um, p);
  const lim = Math.max(sp.dx, sp.dy) * 1.7;
  const toX = v => W/2 + v/lim*(W/2-10);
  const toY = v => H/2 - v/lim*(H/2-10);

  // crosshairs
  ctx.strokeStyle='#ccc'; ctx.lineWidth=0.6; ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(W/2,0); ctx.lineTo(W/2,H); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0,H/2); ctx.lineTo(W,H/2); ctx.stroke();
  ctx.setLineDash([]);

  // reference ellipse (focus)
  const rx0=p.d0x*1e9/lim*(W/2-10), ry0=p.d0y*1e9/lim*(H/2-10);
  ctx.strokeStyle=GR; ctx.lineWidth=1; ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.ellipse(W/2,H/2,rx0,ry0,0,0,2*Math.PI); ctx.stroke();
  ctx.setLineDash([]);

  // actual ellipse
  const rx=sp.dx/lim*(W/2-10), ry=sp.dy/lim*(H/2-10);
  ctx.save(); ctx.globalAlpha=0.15; ctx.fillStyle=R;
  ctx.beginPath(); ctx.ellipse(W/2,H/2,rx,ry,0,0,2*Math.PI); ctx.fill(); ctx.restore();
  ctx.strokeStyle=R; ctx.lineWidth=2;
  ctx.beginPath(); ctx.ellipse(W/2,H/2,rx,ry,0,0,2*Math.PI); ctx.stroke();

  // dimension arrows
  arrow(ctx, W/2-rx, H/2, W/2+rx, H/2, B);
  ctx.font='bold 11px sans-serif'; ctx.fillStyle=B; ctx.textAlign='center';
  ctx.fillText(`d_x = ${sp.dx.toFixed(1)} nm`, W/2, H/2+ry+16);

  arrow(ctx, W/2, H/2-ry, W/2, H/2+ry, T);
  ctx.font='bold 11px sans-serif'; ctx.fillStyle=T; ctx.textAlign='left';
  ctx.fillText(`d_y`, W/2+rx+4, H/2-4);
  ctx.fillText(`${sp.dy.toFixed(1)} nm`, W/2+rx+4, H/2+10);

  // labels
  ctx.font='10px sans-serif'; ctx.fillStyle=GR; ctx.textAlign='center';
  ctx.fillText('-- d₀ at focus', W/2, H-6);
}

// ── Math box ──────────────────────────────────────────────────────────────
function updateMath(dz_um, sx, sy, p) {
  const sp = spotSize(dz_um, p);
  const dz = Math.abs(dz_um)*1e-6;
  const ageo = dz>5e-8 ? (sp.cbx/2*1e-9)/dz : p.ax;
  const okX = Math.abs(dz_um) <= p.dofx*1e6;
  const okY = Math.abs(dz_um) <= p.dofy*1e6;
  const reg = dz_um<0?'before focus':dz_um>0?'past focus':'AT FOCUS';
  const f = (v,n=3) => v.toFixed(n);

  document.getElementById('mathBox').textContent =
`┌─ Slit → beam params ──────────┐

  X slit = ${f(sx,0)} µm (ref ${SX_REF} µm)
  Y slit = ${f(sy,0)} µm (ref ${SY_REF} µm)

  α_x = α_ref × (sx/sx_ref)
      = ${f(AX_REF*1e3)} × (${f(sx,0)}/${SX_REF})
      = ${f(p.ax*1e3)} mrad

  d₀_x = ε_x / α_x
       = ${f(p.d0x*1e9,2)} nm

  α_y  = ${f(p.ay*1e3)} mrad
  d₀_y = ${f(p.d0y*1e9,2)} nm

────────────────────────────────

  Δz = ${f(dz_um,2)} µm  (${reg})

  Cone blur X = ${f(sp.cbx,2)} nm
  Cone blur Y = ${f(sp.cby,2)} nm

  d_x = √(${f(p.d0x*1e9,2)}² + ${f(sp.cbx,2)}²)
      = ${f(sp.dx,2)} nm

  d_y = ${f(sp.dy,2)} nm

────────────────────────────────

  DoF_x = ${f(p.dofx*1e6,3)} µm
          ${okX?'INSIDE ':'OUTSIDE ✗'}

  DoF_y = ${f(p.dofy*1e6,3)} µm
          ${okY?'INSIDE ':'OUTSIDE ✗'}

└────────────────────────────────┘`;
}

// ── Main update ───────────────────────────────────────────────────────────
function update() {
  const dz = parseFloat(document.getElementById('slDz').value);
  const sx = parseFloat(document.getElementById('slSx').value);
  const sy = parseFloat(document.getElementById('slSy').value);

  document.getElementById('vlDz').textContent = dz.toFixed(1);
  document.getElementById('vlSx').textContent = sx.toFixed(0);
  document.getElementById('vlSy').textContent = sy.toFixed(0);

  const p = beamParams(sx, sy);
  drawSide(dz, p);
  drawFront(dz, p);
  updateMath(dz, sx, sy, p);
}

['slDz','slSx','slSy'].forEach(id =>
  document.getElementById(id).addEventListener('input', update));

update();
</script>
</body>
</html>
  
| Parameter | Value |
|---|---|
| Accelerator | 3.5 MV Singletron (HVEE) |
| Beam energy | 2 MeV protons |
| Objective aperture | 8 × 4 µm² |
| Beam half-divergence | ~3 µrad |
| Lens configuration | Spaced Oxford triplet (CDC) |
| Object-to-lens distance | 7.5 m |
| Image distance | 30 mm |
| Demagnification (X) | 857 |
| Demagnification (Y) | 130 |
| Quadrupole power supply resolution | 2 ppm (Bruker) |

The beam spot size plays a critical role in controlling patterning precision. A larger spot effectively broadens the dose profile at the feature edge, increasing the measured FWHM f and reducing the achievable sidewall angle. The relationship is non-linear: when the beam is already well focused, small changes in spot size have little effect on θ, but when the beam is defocused or the focal plane is misplaced, the degradation in θ becomes significant. For this reason, beam focus is verified before each writing session by scanning across the free-standing resolution standard and fitting the transmitted or secondary electron signal to an error function to extract the beam FWHM [4].

The focal plane position can be physically adjusted by moving the sample stage along the beam axis with an accuracy of approximately 1 µm. Since the beam converges to a minimum spot at the focal point and diverges either side, placing the sample above or below focus increases
the effective spot size at the resist surface according to the cone half-angle α and the defocus distance Δz. Systematic variation of the focal plane position during writing is therefore a potential strategy for compensating residual beam divergence at depth, keeping the beam optimally focused throughout the resist thickness rather than only at the surface.

### 3.5 P-beam energy dosage 
In proton-beam writing, dose refers to the total charge delivered per unit area of resist, expressed in nC/mm². It is the product of the beam current, the dwell time per pixel, and the inverse of the pixel area. Physically, it represents the number of protons that have passed through each unit area of the resist surface, a higher dose means more protons, more secondary electron generation, and therefore more chain scission events per unit volume of PMMA.

Dose is distinct from energy. Energy determines *where* in the resist the protons stop and how deeply they penetrate. Dose determines *how much* chemical damage is accumulated at each depth along that path.

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/dosage.png" alt="resolution fabrication overview"  style="margin: 5px;">
  
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 3.5:</strong> dosage and development testing
  </figcaption>
</figure>
 
Proton beam writing doses ranged from 75–175 nC/mm².

The minimum dose required to fully develop the exposed PMMA is called the threshold dose or clearing dose. Below this value, the chain scission density is insufficient for the developer (DI:IPA 7:3) to dissolve the exposed material, and the feature will either partially develop or not develop at all as seen below.

![alt text](images/dev.png)

#### Effects of dosage

**Underdose (below ~50–75 nC/mm²):** Insufficient chain scission. The exposed PMMA
does not reach its gel point , the minimum molecular weight reduction required for the developer to dissolve it. The feature either does not develop, or develops incompletely, leaving a residual PMMA layer at the bottom of the trench. This prevents metal from reaching the substrate and produces no grid feature after lift-off.

**Correct dose (> 75nC/mm²):** The exposed volume is cleanly dissolved by the developer from top to bottom, producing a well-defined trench with vertical sidewalls whose quality is limited by the lateral straggle of the beam, as characterised in
Section 3.2.

**Overdose (above > 280 nC/mm²):** Excess secondary electron generation begins to expose resist beyond the intended beam boundary. The trench widens beyond the written pattern, reducing the effective CD and degrading sidewall verticality. At very high doses the PMMA surface can blister or form bubbles due to rapid outgassing of volatile chain scission products, permanently damaging the resist structure.

**Extreme overdose (above ~3.5 × 10¹⁴ ions/cm²):** PMMA undergoes a positive-to-negative resist transition, and the exposed regions become
insoluble rather than soluble.  This regime is not relevant to the present project but would fundamentally change the development polarity if accidentally reached.


### 3.6 Metal deposition characteristics

| Material | Deposition technique | Melting point (°C) | Conductivity (S/m) | Reasoning |
|---|---|---|---|------|
| Au | Magnetron sputtering | 1064 | 4.52 × 10⁷ |   gives excellent SEM/TEM contrast; chemically inert; well-established PVD process; lift-off compatible |
| Pd | E-beam evaporation | 1554.9 | 9.5 × 10⁶ |  good contrast; chemically stable; used for X-ray zone plates and resolution standards; higher melting point limits substrate heating risk |
| Cr | Magnetron sputtering |  1907| 7.9 × 10⁶ |  Deposited as adhesion buffer layer beneath Au/Pd; strong bonding to Si oxide|
| DLC | FCVA | N/A (amorphous) | ~10⁻³–10² (sp²/sp³ dependent)  | smoother surface|

The metals selected for this project were chosen on the basis of the criteria established
in Section 2.4 , lift-off compatibility, electron scattering contrast, chemical stability, and lattice mismatch, alongside the practical constraint of cleanroom availability at CIBA.

Gold (Au) was selected as the primary structural metal due to its well-established compatibility with magnetron sputtering workflows at CIBA, hasa potentially high electron detector backscatter contrast, and its chemical inertness. Chromium (Cr) was included as an adhesion buffer layer beneath Au, exploiting its strong bonding to native silicon oxide and its ability to reduce internal stress arising from the Au–Si lattice mismatch. Palladium (Pd) was evaluated as an alternative primary metal good chemical stability, and a higher melting point than Au, reducing the risk of substrate heating during e-beam evaporation. Diamond-like carbon (DLC) was investigated as a candidate surface coating to improve roughness performance after surface concerns with sputtered Au were observed. Titanium (Ti) was later introduced as an alternative adhesion layer to Cr, offering improved interfacial bonding without the additional conductivity and optical contrast change associated with Cr.

### 3.7 Fabricated samples composition

| Sample | Cr | Pd | Au | DLC | Ti |
|---|:---:|:---:|:---:|:---:|:---:|
| 1 ,Au/Cr/Si | 2nm | |  40nm | | |
| 2 ,DLC/Si |  | | | 10nm  |
| 3 ,DLC/Pd/Si |  |2nm  | |10nm |
| 4 ,Au/Pd/Si |  | 2nm |  | |
| 5 - DLC/Pd/Ti/Si| | | | |2nm|
| 6 - Pd/Ti/Si| | 40nm | | | 2nm |


[<--Prev: Methodology ](Methology.md) | [Next: Results and analysis →](fna.md)


### References

<div class="references">

<ol>
  <li>Microchem / Kayaku Advanced Materials, "PMMA Data Sheet," 2019. Available:
      <a href="https://kayakuam.com/wp-content/uploads/2019/09/PMMA_Data_Sheet.pdf">kayakuam.com</a></li>

  <li>J. A. van Kan, P. Malar, and A. B. H. Tay, "Resist materials for proton beam writing:
      a review," <em>Applied Surface Science</em>, 2014.
      DOI: <a href="https://doi.org/10.1016/j.apsusc.2014.04.147">10.1016/j.apsusc.2014.04.147</a></li>

  <li>University of Chicago Pritzker Nanofab, "NANO 495 PMMA process," 2024. Available:
      <a href="https://pnf.uchicago.edu/process/detail/950pmma-a4/">pnf.uchicago.edu</a></li>

  <li>S. Raman, Y. Yao, and J. A. van Kan, "Automatic beam focusing in the 2nd generation
      PBW line at sub-10 nm line resolution," <em>Nuclear Instruments and Methods in Physics
      Research Section B</em>, vol. 348, pp. 22–26, 2015.
      DOI: <a href="https://doi.org/10.1016/j.nimb.2014.12.066">10.1016/j.nimb.2014.12.066</a></li>

  <li>J. A. van Kan, P. Malar et al., "Proton beam writing nanoprobe facility design and
      first test results," <em>Nuclear Instruments and Methods in Physics Research
      Section A</em>, 2011.
      DOI: <a href="https://doi.org/10.1016/j.nima.2010.12.011">10.1016/j.nima.2010.12.011</a></li>
</ol>

</div>
