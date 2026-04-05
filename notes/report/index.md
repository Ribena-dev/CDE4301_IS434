---
layout: default
title: "Abstract"
---

<div style="text-align: center; padding: 2rem 0 1rem;">
  <img src="images/NUS_logo.png" alt="NUS Logo" style="height: 60px; display: inline-block; margin: 0 auto 1rem;">
  <h1 style="font-size: clamp(20px, 3vw, 28px); font-weight: 700; margin-bottom: 0.4rem; text-align: center;">
    Fabrication of a Grid Resolution Standard via Proton-Beam Writing
  </h1>
  <p style="font-size: 14px; color: #555; text-align: center; margin-bottom: 0.2rem;">
    CDE4301 Innovation &amp; ·&nbsp; AY2025
  </p>
  <p style="font-size: 14px; color: #555; text-align: center; margin-bottom: 0.2rem;">
    Devinaa Kumeresh 
  </p>
  <p style="font-size: 14px; color: #555; text-align: center; margin-bottom: 0.2rem;">
    A0266490X
  </p>


</div>

<hr style="border: none; border-top: 2px solid #1a1a1a; margin: 1.5rem 0;">

## Acknowledgements 

## Abstract



<hr style="border: none; border-top: 1px solid #ddd; margin: 2rem 0;">

## Table of Contents

<div id="toc-home" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 1rem;">
    {% include toc_card.html
     chapter="1"
     title="Introduction"
     path="Introduction.md"
     url="Introduction.html" %}

  {% include toc_card.html
     chapter="2"
     title="Methodology"
     path="Methology.md"
     url="Methology.html" %}

  {% include toc_card.html
     chapter="3"
     title="Fabrication"
     path="Fabrication.md"
     url="Fabrication.html" %}

  {% include toc_card.html
     chapter="4"
     title="Results & Analysis"
     path="fna.md"
     url="fna.html" %}

  {% include toc_card.html
     chapter="5"
     title="Future Work"
     path="FW.md"
     url="FW.html" %}



  </div>
</div>

<style>
.toc-card:hover {
  border-color: #1a1a1a !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
</style>

<script>
const chapters = [
  {
    url: '/CDE4301_IS434/notes/report/Introduction.html',
    targetId: 'toc-intro-links'
  },
  {
    url: '/CDE4301_IS434/notes/report/Methology.html',
    targetId: 'toc-meth-links'
  },
  {
    url: '/CDE4301_IS434/notes/report/Fabrication.html',
    targetId: 'toc-fab-links'
  },
  {
    url: '/CDE4301_IS434/notes/report/fna.html',
    targetId: 'toc-fna-links'
  },
  {
    url: '/CDE4301_IS434/notes/report/FW.html',
    targetId: 'toc-fw-links'
  }
];

chapters.forEach(ch => {
  fetch(ch.url)
    .then(r => r.text())
    .then(html => {
      const parser = new DOMParser();
      const doc    = parser.parseFromString(html, 'text/html');
      const headings = doc.querySelectorAll('h2, h3');
      const container = document.getElementById(ch.targetId);
      if(!container) return;

      let items = [];
      headings.forEach(h => {
        const id   = h.id || h.textContent.toLowerCase()
                       .replace(/[^a-z0-9\s]/g,'')
                       .trim().replace(/\s+/g,'-');
        const indent = h.tagName === 'H3' ? 'padding-left:0.8rem;' : '';
        items.push(
          `<div style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;${indent}">`
          + `<a href="${ch.url}#${id}" style="color:#555;text-decoration:none;">`
          + `${h.textContent}</a></div>`
        );
      });

      container.innerHTML = items.slice(0, 6).join('');
      if(items.length > 6){
        container.innerHTML +=
          `<div style="color:#aaa;font-size:11px;margin-top:2px;">+${items.length - 6} more sections</div>`;
      }
    })
    .catch(() => {});
});
</script>

<hr style="border: none; border-top: 1px solid #ddd; margin: 2.5rem 0;">


