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


<style>
  .sw-wrap { font-family: var(--font-sans); padding: 1rem 0; }
  .sw-canvases { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 1.25rem; }
  .sw-panel-label { font-size: 11px; font-weight: 500; color: var(--color-text-secondary); margin-bottom: 6px; }
  canvas { display: block; width: 100%; background: var(--color-background-primary); border: 0.5px solid var(--color-border-tertiary); border-radius: 8px; }
  .sw-sliders { display: flex; flex-direction: column; gap: 10px; margin-bottom: 1.25rem; }
  .sw-row { display: flex; align-items: center; gap: 10px; }
  .sw-row label { font-size: 12px; color: var(--color-text-secondary); width: 160px; flex-shrink: 0; }
  .sw-row input { flex: 1; }
  .sw-row .vout { font-size: 13px; font-weight: 500; width: 60px; text-align: right; color: var(--color-text-primary); }
  .sw-metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
  .sw-card { background: var(--color-background-secondary); border-radius: 8px; padding: 10px 12px; }
  .sw-card .lbl { font-size: 11px; color: var(--color-text-secondary); margin-bottom: 4px; }
  .sw-card .num { font-size: 20px; font-weight: 500; color: var(--color-text-primary); }
  .sw-card.warn .num { color: #A32D2D; }
  .sw-card.ok .num { color: #0F6E56; }
</style>

<div class="sw-wrap">
  <div class="sw-canvases">
    <div>
      <div class="sw-panel-label">Feature cross-section (CD-SEM view)</div>
      <canvas id="cvXsec" width="400" height="280"></canvas>
    </div>
    <div>
      <div class="sw-panel-label">Simulated SEM intensity profile</div>
      <canvas id="cvSEM" width="400" height="280"></canvas>
    </div>
  </div>

  <div class="sw-sliders">
    <div class="sw-row">
      <label>Sidewall angle (°)</label>
      <input type="range" id="slSWA" min="60" max="90" step="0.5" value="85">
      <span class="vout" id="vlSWA">85°</span>
    </div>
    <div class="sw-row">
      <label>Feature height h (nm)</label>
      <input type="range" id="slH" min="10" max="80" step="1" value="18">
      <span class="vout" id="vlH">18 nm</span>
    </div>
    <div class="sw-row">
      <label>Nominal CD (nm)</label>
      <input type="range" id="slCD" min="8" max="40" step="1" value="16">
      <span class="vout" id="vlCD">16 nm</span>
    </div>
  </div>

  <div class="sw-metrics">
    <div class="sw-card">
      <div class="lbl">deviation from 90°</div>
      <div class="num" id="mDev">5.0°</div>
    </div>
    <div class="sw-card">
      <div class="lbl">CD error (nm)</div>
      <div class="num" id="mErr">3.1 nm</div>
    </div>
    <div class="sw-card" id="mcPct">
      <div class="lbl">CD error (%)</div>
      <div class="num" id="mPct">19.6%</div>
    </div>
    <div class="sw-card">
      <div class="lbl">W bottom (nm)</div>
      <div class="num" id="mWbot">19.1 nm</div>
    </div>
  </div>
</div>

<script>
const BLUE='#185FA5', TEAL='#0F6E56', RED='#A32D2D', AMB='#BA7517', GR='#888780';

function gauss(x, mu, sig) {
  return Math.exp(-0.5*((x-mu)/sig)**2);
}

function drawXsec(swa, h, cdTop) {
  const cv = document.getElementById('cvXsec');
  const ctx = cv.getContext('2d');
  const W=cv.width, H=cv.height;
  ctx.clearRect(0,0,W,H);

  const PAD=40, GND=H-50;
  const delta = (90-swa)*Math.PI/180;
  const cdBot = cdTop + 2*h*Math.tan(delta);
  const scale = (W-2*PAD) / Math.max(cdBot*2.5, 60);

  const cx = W/2;
  const yTop = GND - h*scale;
  const yBot = GND;
  const xTopL = cx - (cdTop/2)*scale;
  const xTopR = cx + (cdTop/2)*scale;
  const xBotL = cx - (cdBot/2)*scale;
  const xBotR = cx + (cdBot/2)*scale;

  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--color-background-primary') || '#fff';

  ctx.strokeStyle='#ccc'; ctx.lineWidth=0.5; ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(0,GND); ctx.lineTo(W,GND); ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle='#B5D4F4'; ctx.strokeStyle=BLUE; ctx.lineWidth=1.5;
  ctx.beginPath();
  ctx.moveTo(xBotL, yBot);
  ctx.lineTo(xTopL, yTop);
  ctx.lineTo(xTopR, yTop);
  ctx.lineTo(xBotR, yBot);
  ctx.closePath();
  ctx.fill(); ctx.stroke();

  ctx.fillStyle='#D3D1C7';
  ctx.fillRect(PAD-10, GND, W-2*PAD+20, 18);
  ctx.strokeStyle='#888'; ctx.lineWidth=1;
  ctx.strokeRect(PAD-10, GND, W-2*PAD+20, 18);

  const isDark = window.matchMedia('(prefers-color-scheme:dark)').matches;
  const textCol = isDark ? '#e0ddd6' : '#333';

  ctx.font='11px sans-serif'; ctx.fillStyle=textCol;

  ctx.setLineDash([3,3]); ctx.strokeStyle=RED; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(xTopL,yTop-6); ctx.lineTo(xTopL,yBot+20); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(xTopR,yTop-6); ctx.lineTo(xTopR,yBot+20); ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle=RED; ctx.lineWidth=1.5;
  const ay=yTop-14;
  ctx.beginPath(); ctx.moveTo(xTopL,ay); ctx.lineTo(xTopR,ay); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(xTopL,ay-4); ctx.lineTo(xTopL,ay+4); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(xTopR,ay-4); ctx.lineTo(xTopR,ay+4); ctx.stroke();
  ctx.fillStyle=RED; ctx.textAlign='center';
  ctx.fillText(`W_top = ${cdTop.toFixed(0)} nm (SEM measured)`, cx, ay-6);

  if(Math.abs(delta)>0.005){
    ctx.setLineDash([2,3]); ctx.strokeStyle=TEAL; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(xBotL,yBot+24); ctx.lineTo(xBotR,yBot+24); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=TEAL; ctx.textAlign='center';
    ctx.fillText(`W_bot = ${cdBot.toFixed(1)} nm (true CD)`, cx, yBot+36);
  }

  ctx.strokeStyle=AMB; ctx.lineWidth=1; ctx.setLineDash([]);
  const hx=xTopR+12;
  ctx.beginPath(); ctx.moveTo(hx,yTop); ctx.lineTo(hx,yBot); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(hx-3,yTop); ctx.lineTo(hx+3,yTop); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(hx-3,yBot); ctx.lineTo(hx+3,yBot); ctx.stroke();
  ctx.fillStyle=AMB; ctx.textAlign='left';
  ctx.fillText(`h = ${h} nm`, hx+5, (yTop+yBot)/2+4);

  ctx.fillStyle=BLUE; ctx.textAlign='center';
  ctx.font='10px sans-serif';
  ctx.fillText(`SWA = ${swa}°`, xTopL-18, (yTop+yBot)/2+4);
  const angLen=20;
  ctx.strokeStyle=BLUE; ctx.lineWidth=1.2;
  ctx.beginPath();
  ctx.arc(xBotL, yBot, angLen, -Math.PI/2, -Math.PI/2+delta, false);
  ctx.stroke();

  ctx.font='bold 11px sans-serif'; ctx.fillStyle=BLUE; ctx.textAlign='left';
  ctx.fillText('Silicon substrate', PAD, GND+14);

  ctx.font='11px sans-serif'; ctx.fillStyle=BLUE; ctx.textAlign='center';
  ctx.fillText('Metal feature', cx, (yTop+yBot)/2+4);
}

function drawSEM(swa, h, cdTop) {
  const cv = document.getElementById('cvSEM');
  const ctx = cv.getContext('2d');
  const W=cv.width, H=cv.height;
  ctx.clearRect(0,0,W,H);

  const isDark = window.matchMedia('(prefers-color-scheme:dark)').matches;
  const textCol = isDark ? '#e0ddd6' : '#333';

  const delta = (90-swa)*Math.PI/180;
  const cdBot = cdTop + 2*h*Math.tan(delta);
  const PAD=40, plotH=H-70;

  const xRange = Math.max(cdBot*3, 60);
  const xL_top = W/2 - (cdTop/2)/xRange*(W-2*PAD);
  const xR_top = W/2 + (cdTop/2)/xRange*(W-2*PAD);
  const xL_bot = W/2 - (cdBot/2)/xRange*(W-2*PAD);
  const xR_bot = W/2 + (cdBot/2)/xRange*(W-2*PAD);

  const sigma = (W-2*PAD)*2.5/xRange;

  const N=300;
  const xs=[], ys=[];
  for(let i=0;i<N;i++){
    const px = PAD + i/(N-1)*(W-2*PAD);
    const x_nm = (px - W/2) / (W-2*PAD) * xRange;
    const x_top_l = -cdTop/2, x_top_r = cdTop/2;

    let sig=0;
    sig += 0.25;
    if(x_nm > -cdBot/2 && x_nm < cdBot/2) sig += 0.30;

    const sig_nm = 3.0;
    const sig_px = sig_nm/(xRange)*(W-2*PAD);
    sig += 0.9*gauss(px, xL_top, sig_px);
    sig += 0.9*gauss(px, xR_top, sig_px);

    const sig_bot_px = sig_nm*1.4/(xRange)*(W-2*PAD);
    if(swa < 89){
      sig += 0.35*gauss(px, xL_bot, sig_bot_px);
      sig += 0.35*gauss(px, xR_bot, sig_bot_px);
    }
    xs.push(px);
    ys.push(sig);
  }

  const yMin=0, yMax=1.6;
  const toY = v => plotH + 30 - (v-yMin)/(yMax-yMin)*plotH;

  ctx.strokeStyle='#ddd'; ctx.lineWidth=0.5;
  for(let g=0;g<=4;g++){
    const gy=toY(g*0.4);
    ctx.beginPath(); ctx.moveTo(PAD,gy); ctx.lineTo(W-PAD,gy); ctx.stroke();
  }

  ctx.strokeStyle=BLUE; ctx.lineWidth=2;
  ctx.beginPath();
  for(let i=0;i<N;i++){
    i===0 ? ctx.moveTo(xs[i],toY(ys[i])) : ctx.lineTo(xs[i],toY(ys[i]));
  }
  ctx.stroke();

  ctx.strokeStyle=RED; ctx.lineWidth=1.5; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(xL_top,30); ctx.lineTo(xL_top,plotH+30); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(xR_top,30); ctx.lineTo(xR_top,plotH+30); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle=RED; ctx.font='10px sans-serif'; ctx.textAlign='center';
  ctx.fillText('← SEM measured edges →', W/2, 22);

  const arY=plotH+42;
  ctx.strokeStyle=RED; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(xL_top,arY); ctx.lineTo(xR_top,arY); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(xL_top,arY-3); ctx.lineTo(xL_top,arY+3); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(xR_top,arY-3); ctx.lineTo(xR_top,arY+3); ctx.stroke();
  ctx.fillStyle=RED; ctx.textAlign='center';
  ctx.fillText(`W_top = ${cdTop.toFixed(0)} nm`, W/2, arY+12);

  if(Math.abs(delta)>0.005 && cdBot>cdTop+0.5){
    ctx.strokeStyle=TEAL; ctx.lineWidth=1.2; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(xL_bot,30); ctx.lineTo(xL_bot,plotH+30); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(xR_bot,30); ctx.lineTo(xR_bot,plotH+30); ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle=TEAL; ctx.lineWidth=1.2;
    const arY2=plotH+56;
    ctx.beginPath(); ctx.moveTo(xL_bot,arY2); ctx.lineTo(xR_bot,arY2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(xL_bot,arY2-3); ctx.lineTo(xL_bot,arY2+3); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(xR_bot,arY2-3); ctx.lineTo(xR_bot,arY2+3); ctx.stroke();
    ctx.fillStyle=TEAL; ctx.textAlign='center';
    ctx.fillText(`W_bot = ${cdBot.toFixed(1)} nm`, W/2, arY2+12);
  }

  ctx.font='11px sans-serif'; ctx.fillStyle=textCol;
  ctx.textAlign='left'; ctx.fillText('SE intensity (a.u.)', PAD, 24);
  ctx.textAlign='center'; ctx.fillText('scan position', W/2, H-4);
}

function update() {
  const swa = parseFloat(document.getElementById('slSWA').value);
  const h   = parseFloat(document.getElementById('slH').value);
  const cd  = parseFloat(document.getElementById('slCD').value);

  document.getElementById('vlSWA').textContent = swa.toFixed(1)+'°';
  document.getElementById('vlH').textContent   = h.toFixed(0)+' nm';
  document.getElementById('vlCD').textContent  = cd.toFixed(0)+' nm';

  const delta = (90-swa)*Math.PI/180;
  const errNm = 2*h*Math.tan(delta);
  const errPct = errNm/cd*100;
  const wBot = cd + errNm;

  document.getElementById('mDev').textContent  = (90-swa).toFixed(1)+'°';
  document.getElementById('mErr').textContent  = errNm.toFixed(1)+' nm';
  document.getElementById('mPct').textContent  = errPct.toFixed(1)+'%';
  document.getElementById('mWbot').textContent = wBot.toFixed(1)+' nm';

  const pctCard = document.getElementById('mcPct');
  pctCard.className = 'sw-card' + (errPct>15?' warn':errPct<5?' ok':'');

  drawXsec(swa, h, cd);
  drawSEM(swa, h, cd);
}

['slSWA','slH','slCD'].forEach(id=>
  document.getElementById(id).addEventListener('input',update));
update();
</script>


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
 

