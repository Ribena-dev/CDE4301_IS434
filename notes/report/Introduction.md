## Introduction 


### 1.1 Overview of Resolution Standards
Driven by Moore's Law, which predicts the doubling of transistor density approximately every two years, semiconductor feature sizes have scaled from the micrometre range in the 1970s to sub-2 nm nodes in commercial production today [1] . This relentless miniaturisation has rendered conventional optical microscopy impractical for surface characterisation; the wavelength of visible light (380–700 nm) fundamentally limits optical resolution to length scales far exceeding those of modern device features [2]. Consequently, there is a pressing need for higher-precision characterisation instruments.

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/Scaling-of-transistor-size-physical-gate-length-L-g-to-sustain-Moores-Law.png" alt="moores law tranistor gate scalling" >
  
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 1.1:</strong> Scaling of transistor size physical-gate length to-sustain Moore's Law
  </figcaption>
</figure>

To meet this need, researchers and manufacturers rely on a suite of advanced metrology instruments capable of characterising nanometre and sub-nanometre scale features. These include scanning electron microscopes (SEM), critical dimension atomic force microscopes (CD-AFM), transmission electron microscopes (TEM), and extreme ultraviolet (EUV) scatterometry systems. CD-AFM measures surface features by dragging a calibrated flared tip across a surface, like a record player needle tracing a groove, with potential width uncertainties as low as 1 nm  [3]. EUV scatterometry illuminates a patterned surface with extreme ultraviolet light (wavelength ~13.5 nm) and reconstructs the three-dimensional profile of surface features by analysing the angular distribution of scattered intensity, enabling non-destructive characterisation of line width, sidewall angle, and surface roughness at the sub-10 nm scale [4]. However, the accuracy of measurements from any such instrument depends entirely on the quality of its calibration [3] — which is where resolution and calibration standards become essential.

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/tin_nm.png" alt="Tin sphere resolution standards" width="280" style="margin: 5px;">
  <img src="images/grid_nm.png" alt="Grid resolution standards" width="280" style="margin: 5px;">
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 1.2:</strong> Common resolution standards: tin spheres (left) and nano-grids (right)
  </figcaption>
</figure>

### 1.2 The Importance of Sidewall Angles in Grid Resolution Standards

For instruments such as CD-AFM, 3D-AFM and electron microscopy systems (SEM, EUV), the perpendicularity of the sidewall angle of patterned features is of critical importance to measurement accuracy.
In CD-AFM, a vertical parallel structure (VPS), is required as the primary tip characteriser because it allows measurement of the CD tip width independently of the specific flare geometry of the probe. The calibration relies on the sidewalls being vertical: the finer details of the tip-sample interaction, including feature sidewall angle and corner radius, introduce higher-order tip effects that cause systematic biases in measured linewidth.Any deviation of the reference sidewall from 90° introduces an uncharacterised geometric bias into every subsequent measurement the instrument makes. [5] [3] [6] [7]

<!-- <iframe
  src="scripts/sidewall_angle_cd_error.html" 
  width="100%" 
  height="580px" 
  style="border:none; border-radius:6px;"
  sandbox="allow-scripts" >
</iframe> -->


<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Sidewall angle — SEM scan model</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: var(--font-sans, sans-serif); padding: 1rem 0; }
  .canvases { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 1.25rem; }
  .lbl { font-size: 11px; font-weight: 500; color: var(--color-text-secondary); margin-bottom: 6px; }
  canvas { display: block; width: 100%; background: var(--color-background-primary, #fff); border: 0.5px solid var(--color-border-tertiary); border-radius: 8px; }
  .sliders { display: flex; flex-direction: column; gap: 10px; margin-bottom: 1.25rem; }
  .row { display: flex; align-items: center; gap: 10px; }
  .row label { font-size: 12px; color: var(--color-text-secondary); width: 160px; flex-shrink: 0; }
  .row input { flex: 1; }
  .row .v { font-size: 13px; font-weight: 500; width: 60px; text-align: right; color: var(--color-text-primary); }
  .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  .card { background: var(--color-background-secondary); border-radius: 8px; padding: 10px 12px; }
  .card .lbl2 { font-size: 11px; color: var(--color-text-secondary); margin-bottom: 4px; }
  .card .num { font-size: 20px; font-weight: 500; color: var(--color-text-primary); }
  .card.warn .num { color: #A32D2D; }
  .card.ok   .num { color: #0F6E56; }
</style>
</head>
<body>

<div class="canvases">
  <div>
    <div class="lbl">Feature cross-section</div>
    <canvas id="cvX" width="360" height="260"></canvas>
  </div>
  <div>
    <div class="lbl">Simulated SEM intensity profile</div>
    <canvas id="cvS" width="360" height="260"></canvas>
  </div>
</div>

<div class="sliders">
  <div class="row">
    <label style="color:#a32d2d">&#9632; Sidewall angle (°)</label>
    <input type="range" id="sA" min="60" max="90" step="0.5" value="85">
    <span class="v" id="vA" style="color:#a32d2d">85.0°</span>
  </div>
</div>

<div class="metrics">
  <div class="card"><div class="lbl2">deviation from 90°</div><div class="num" id="mD">5.0°</div></div>
  <div class="card"><div class="lbl2">CD error (nm)</div><div class="num" id="mE">—</div></div>
  <div class="card" id="mPc"><div class="lbl2">CD error (%) at 20 nm CD</div><div class="num" id="mP">—</div></div>
</div>

<script>
const H_NM = 40;
const CD   = 20;
const B='#185fa5', T='#0f6e56', R='#a32d2d', A='#ba7517';

// Fixed scale — 1 nm = this many px. Chosen so the feature fills ~40% of canvas width.
// Canvas is 360px wide, PAD=40 each side → usable 280px. Max bot at 60° = 16+2*40*tan30 ≈ 62nm.
// We want 62nm to fit comfortably → SC = 280/90 ≈ 3.1
const PAD = 40;
const SC  = 2.8;  // px per nm — FIXED, never changes

function gauss(x,mu,s){ return Math.exp(-0.5*((x-mu)/s)**2); }

function drawX(swa) {
  const cv=document.getElementById('cvX'), ctx=cv.getContext('2d');
  const W=cv.width, H=cv.height;
  ctx.clearRect(0,0,W,H);

  const d=(90-swa)*Math.PI/180;
  const GND=H-48;
  const cx=W/2;

  // Fixed top corners
  const xTL=cx-(CD/2)*SC, xTR=cx+(CD/2)*SC;
  const yT=GND-H_NM*SC, yB=GND;

  // Base corners slide out with angle
  const overhang=H_NM*Math.tan(d);
  const xBL=cx-(CD/2)*SC-overhang*SC;
  const xBR=cx+(CD/2)*SC+overhang*SC;

  // substrate
  ctx.strokeStyle='#e0ddd6'; ctx.lineWidth=0.5; ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(0,yB); ctx.lineTo(W,yB); ctx.stroke();
  ctx.setLineDash([]);

  // feature trapezoid
  ctx.fillStyle='#b5d4f4'; ctx.strokeStyle=B; ctx.lineWidth=1.5;
  ctx.beginPath();
  ctx.moveTo(xBL,yB); ctx.lineTo(xTL,yT);
  ctx.lineTo(xTR,yT); ctx.lineTo(xBR,yB);
  ctx.closePath();
  ctx.fill(); ctx.stroke();

  // substrate bar
  ctx.fillStyle='#d3d1c7'; ctx.strokeStyle='#999'; ctx.lineWidth=1;
  ctx.fillRect(PAD-8,yB,W-2*PAD+16,16);
  ctx.strokeRect(PAD-8,yB,W-2*PAD+16,16);

  ctx.font='11px sans-serif';

  // top edge markers (red — SEM reads these)
  ctx.setLineDash([3,3]); ctx.strokeStyle=R; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(xTL,yT-6); ctx.lineTo(xTL,yB+18); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(xTR,yT-6); ctx.lineTo(xTR,yB+18); ctx.stroke();
  ctx.setLineDash([]);
  const ay=yT-14;
  ctx.strokeStyle=R; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(xTL,ay); ctx.lineTo(xTR,ay); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(xTL,ay-3); ctx.lineTo(xTL,ay+3); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(xTR,ay-3); ctx.lineTo(xTR,ay+3); ctx.stroke();
  ctx.fillStyle=R; ctx.textAlign='center';
  ctx.fillText('W_top — SEM measured', cx, ay-5);

  // bottom edge markers (teal — true CD)
  if(Math.abs(d)>0.005){
    ctx.setLineDash([2,3]); ctx.strokeStyle=T; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(xBL,yB+22); ctx.lineTo(xBR,yB+22); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=T; ctx.textAlign='center';
    ctx.fillText('W_bot — true CD', cx, yB+34);
  }

  // height annotation
  ctx.strokeStyle=A; ctx.lineWidth=1;
  const hx=xTR+14;
  ctx.beginPath(); ctx.moveTo(hx,yT); ctx.lineTo(hx,yB); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(hx-3,yT); ctx.lineTo(hx+3,yT); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(hx-3,yB); ctx.lineTo(hx+3,yB); ctx.stroke();
  ctx.fillStyle=A; ctx.textAlign='left';
  ctx.fillText('h = 40 nm', hx+4, (yT+yB)/2+4);

  // angle arc
  ctx.strokeStyle=B; ctx.lineWidth=1;
  ctx.beginPath(); ctx.arc(xBL,yB,20,-Math.PI/2,-Math.PI/2+d,false); ctx.stroke();
  ctx.fillStyle=B; ctx.textAlign='right';
  ctx.fillText(swa.toFixed(1)+'°', xBL-5, (yT+yB)/2+4);

  ctx.fillStyle='#555'; ctx.textAlign='left'; ctx.fillText('Si substrate', PAD, yB+13);
  ctx.fillStyle=B; ctx.textAlign='center'; ctx.fillText('Metal', cx, (yT+yB)/2+5);
}

function drawS(swa) {
  const cv=document.getElementById('cvS'), ctx=cv.getContext('2d');
  const W=cv.width, H=cv.height;
  ctx.clearRect(0,0,W,H);

  const d=(90-swa)*Math.PI/180;
  const bot=CD+2*H_NM*Math.tan(d);
  const pH=H-60;

  // SEM plot uses a fixed display range tied to the worst-case (60°) so peaks don't jump
  const xRangeNm = CD + 2*H_NM*Math.tan(30*Math.PI/180) + 20; // fixed ~82nm
  const sc=(W-2*PAD)/xRangeNm;

  const xTL=W/2-(CD/2)*sc, xTR=W/2+(CD/2)*sc;
  const xBL=W/2-(bot/2)*sc, xBR=W/2+(bot/2)*sc;
  const sigPx=3*sc;

  const N=300, xs=[], ys=[];
  for(let i=0;i<N;i++){
    const px=PAD+i/(N-1)*(W-2*PAD);
    const nm=(px-W/2)/sc;
    let s=0.22;
    if(nm>-bot/2&&nm<bot/2) s+=0.28;
    s+=0.85*gauss(px,xTL,sigPx);
    s+=0.85*gauss(px,xTR,sigPx);
    if(swa<89){ s+=0.3*gauss(px,xBL,sigPx*1.4); s+=0.3*gauss(px,xBR,sigPx*1.4); }
    xs.push(px); ys.push(s);
  }
  const yMx=1.55, toY=v=>pH+26-(v/yMx)*pH;

  ctx.strokeStyle='#ebe9e2'; ctx.lineWidth=0.5;
  for(let g=0;g<=4;g++){ ctx.beginPath(); ctx.moveTo(PAD,toY(g*0.4)); ctx.lineTo(W-PAD,toY(g*0.4)); ctx.stroke(); }

  ctx.strokeStyle=B; ctx.lineWidth=2;
  ctx.beginPath();
  xs.forEach((x,i)=>i?ctx.lineTo(x,toY(ys[i])):ctx.moveTo(x,toY(ys[i])));
  ctx.stroke();

  // top edge lines — fixed position
  ctx.strokeStyle=R; ctx.lineWidth=1.5; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(xTL,24); ctx.lineTo(xTL,pH+24); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(xTR,24); ctx.lineTo(xTR,pH+24); ctx.stroke();
  ctx.setLineDash([]);
  const ay=pH+36;
  ctx.strokeStyle=R; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(xTL,ay); ctx.lineTo(xTR,ay); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(xTL,ay-3); ctx.lineTo(xTL,ay+3); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(xTR,ay-3); ctx.lineTo(xTR,ay+3); ctx.stroke();
  ctx.fillStyle=R; ctx.textAlign='center'; ctx.font='10px sans-serif';
  ctx.fillText('W_top = '+CD+' nm', W/2, ay+12);

  // bottom edge lines — slide outward
  if(Math.abs(d)>0.005&&bot>CD+0.5){
    ctx.strokeStyle=T; ctx.lineWidth=1.2; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(xBL,24); ctx.lineTo(xBL,pH+24); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(xBR,24); ctx.lineTo(xBR,pH+24); ctx.stroke();
    ctx.setLineDash([]);
    const ay2=pH+50;
    ctx.strokeStyle=T; ctx.lineWidth=1.2;
    ctx.beginPath(); ctx.moveTo(xBL,ay2); ctx.lineTo(xBR,ay2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(xBL,ay2-3); ctx.lineTo(xBL,ay2+3); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(xBR,ay2-3); ctx.lineTo(xBR,ay2+3); ctx.stroke();
    ctx.fillStyle=T; ctx.textAlign='center';
    ctx.fillText('W_bot = '+bot.toFixed(1)+' nm', W/2, ay2+12);
  }

  ctx.fillStyle='#888'; ctx.font='11px sans-serif';
  ctx.textAlign='left'; ctx.fillText('SE intensity', PAD, 18);
  ctx.textAlign='center'; ctx.fillText('scan position →', W/2, H-4);
}

function update(){
  const swa=parseFloat(document.getElementById('sA').value);
  document.getElementById('vA').textContent=swa.toFixed(1)+'°';
  const d=(90-swa)*Math.PI/180;
  const eNm=2*H_NM*Math.tan(d);
  const ePct=eNm/CD*100;
  document.getElementById('mD').textContent=(90-swa).toFixed(1)+'°';
  document.getElementById('mE').textContent=eNm.toFixed(1)+' nm';
  document.getElementById('mP').textContent=ePct.toFixed(1)+'%';
  const pc=document.getElementById('mPc');
  pc.className='card'+(ePct>15?' warn':ePct<5?' ok':'');
  drawX(swa); drawS(swa);
}

document.getElementById('sA').addEventListener('input',update);
update();
</script>
</body>
</html>


In EUV and SEM metrology, the consequences are equally significant. A deviation of just 5° from the ideal 90° sidewall angle has been shown to produce a critical dimension error of up to 20% in a 16 nm line-space pattern [10]. This is because the interaction of the incident beam with a non-vertical sidewall produces asymmetric scattering that systematically shifts the apparent edge position.

### 1.3 Deliverables
This project aims to fabricate a grid resolution standard with a smooth surface finish and a sidewall perpendicularity of 89.9 using proton-beam writing lithography. The use of proton-beam writing, rather than conventional electron-beam lithography, is motivated by the fundamental reduction in lateral beam scattering and will explained later in the report. The fabricated standard will be characterised using CD-AFM and SEM to verify sidewall angle, pitch uniformity, and surface roughness against the defined specifications.

<figure style="text-align: center; margin: 20px 0;">
  <img src="images/grid_dia.png" alt="Grid diagram" width="280" style="margin: 5px;">
  
  <figcaption style="font-style: italic; color: #666; margin-top: 8px; font-size: 14px;">
    <strong>Figure 1.3:</strong> Diagram of grid
  </figcaption>
</figure>

[Next: Methodology →](Methology.md)

### References

<div class="references">
 

 
<ol>
  <li>Z. Yu, S. Tan, R. Han, H. Xiao, and J. He, "Device and technology outlook for 1 nm node and beyond," in <em>Proc. IEEE Int. Conf. Solid-State and Integrated Circuit Technology (ICSICT)</em>, 2004. DOI: 10.1109/ICSICT.2004.1434947</li>
 
  <li>E. Abbe, "Beiträge zur Theorie des Mikroskops und der mikroskopischen Wahrnehmung," <em>Archiv für Mikroskopische Anatomie</em>, vol. 9, pp. 413–468, 1873.</li>
 
  <li>National Institute of Standards and Technology, "Improving CD-AFM measurements from the tip down," NIST News, Mar. 2016. [Online]. Available: <a href="https://www.nist.gov/news-events/news/2016/03/improving-cd-afm-measurements-tip-down">nist.gov</a></li>
 
  <li>I. Pollentier, C.-U. Kim, P. Vandervorst, and E. Hendrickx, "EUV lithography materials characterisation using angle-resolved XPS and EUV scatterometry," <em>physica status solidi (a)</em>, vol. 216, no. 17, 2019. DOI: 10.1002/phvs.201900027</li>
 
  <li>G. Wilkening and L. Koenders, Eds., <em>Nanoscale Calibration Standards and Methods</em>, Part IV. Weinheim: Wiley-VCH, 2005. ISBN: 3-527-40502-X</li>
 
  <li>N. G. Orji, R. G. Dixson, A. Garcia-Gutierrez, B. D. Bunday, and M. Bishop, "Tip characterization method using multi-feature characterizer for CD-AFM," <em>Precision Engineering</em>, 2016. Available: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4803071/">pmc.ncbi.nlm.nih.gov</a></li>
 
  <li>R. G. Dixson, N. G. Orji, J. Fu, and R. Matero, "Lateral tip control effects in CD-AFM metrology: the large tip limit," <em>Journal of Micro/Nanolithography, MEMS, and MOEMS</em>, 2016. Available: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4832421/">pmc.ncbi.nlm.nih.gov</a></li>
 
  <li>K. H. Ko, Y. Moon, C. Jeong, H. Kim, C. U. Jeon, and H. K. Oh, "Influence of a non-ideal sidewall angle of extreme ultra-violet mask absorber for 1×-nm patterning in isomorphic and anamorphic lithography," <em>Microelectronic Engineering</em>, vol. 181, pp. 1–9, 2017. DOI: 10.1016/j.mee.2017.06.007</li>
</ol>
 
</div>
 

