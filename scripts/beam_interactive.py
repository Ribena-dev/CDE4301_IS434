"""
Beam convergence — interactive focal plane explorer
====================================================
Drag the slider to move the sample through z.
The math box, cone diagram, and d(Δz) cursor all update live.

Run:  python beam_interactive.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from matplotlib.patches import FancyArrowPatch

# ── System constants (Table 2-2 and §2.6) ───────────────────────────────────
D0_X  = 9.3e-9    # spot at focus X [m]
D0_Y  = 32e-9     # spot at focus Y [m]
DX    = 857        # demagnification X
DY    = 130        # demagnification Y
A_OBJ = 3e-6      # object divergence half-angle [rad]

# ── Derived once ─────────────────────────────────────────────────────────────
AX    = A_OBJ * DX          # convergence angle X [rad]
AY    = A_OBJ * DY          # convergence angle Y [rad]
DOF_X = D0_X / (2 * AX)    # depth of focus X [m]
DOF_Y = D0_Y / (2 * AY)    # depth of focus Y [m]

BLUE, TEAL, AMBER, RED = '#185FA5', '#0F6E56', '#BA7517', '#A32D2D'

# ── Math ─────────────────────────────────────────────────────────────────────
def calc(dz_um):
    dz   = dz_um * 1e-6
    cx   = 2 * AX * abs(dz)
    cy   = 2 * AY * abs(dz)
    dx   = np.sqrt(D0_X**2 + cx**2)
    dy   = np.sqrt(D0_Y**2 + cy**2)
    return dict(
        dz_um=dz_um,
        ax_mrad=AX*1e3, ay_mrad=AY*1e3,
        cone_x=cx*1e9, cone_y=cy*1e9,
        dx_nm=dx*1e9,  dy_nm=dy*1e9,
        xg=(dx/D0_X-1)*100, yg=(dy/D0_Y-1)*100,
        ok_x=abs(dz)<=DOF_X, ok_y=abs(dz)<=DOF_Y,
    )

# ── Build math string ─────────────────────────────────────────────────────────
def math_text(r):
    ok_x = "YES ✓" if r['ok_x'] else "NO  ✗"
    ok_y = "YES ✓" if r['ok_y'] else "NO  ✗"
    return (
        f"Step 1 — convergence angles\n"
        f"  α_x = {A_OBJ*1e6:.1f} µrad × {DX}  =  {r['ax_mrad']:.3f} mrad\n"
        f"  α_y = {A_OBJ*1e6:.1f} µrad × {DY}  =  {r['ay_mrad']:.3f} mrad\n\n"
        f"Step 2 — cone blur at Δz = {r['dz_um']:+.2f} µm\n"
        f"  2α_x·|Δz|  =  {r['cone_x']:.3f} nm\n"
        f"  2α_y·|Δz|  =  {r['cone_y']:.3f} nm\n\n"
        f"Step 3 — d(Δz) = √(d₀² + (2α·Δz)²)\n"
        f"  d_x  =  √({D0_X*1e9:.1f}²  +  {r['cone_x']:.2f}²)  =  {r['dx_nm']:.2f} nm  [{r['xg']:+.1f}%]\n"
        f"  d_y  =  √({D0_Y*1e9:.1f}²  +  {r['cone_y']:.2f}²)  =  {r['dy_nm']:.2f} nm  [{r['yg']:+.1f}%]\n\n"
        f"Inside DoF_x ({DOF_X*1e6:.2f} µm)?   {ok_x}\n"
        f"Inside DoF_y ({DOF_Y*1e6:.1f} µm)?  {ok_y}"
    )

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('#f8f8f6')

gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    left=0.06, right=0.97,
    top=0.91,  bottom=0.14,
    hspace=0.38, wspace=0.35
)

ax_cone  = fig.add_subplot(gs[0, :2])   # cone diagram — top wide
ax_curve = fig.add_subplot(gs[1, :2])   # d(Δz) curve  — bottom wide
ax_math  = fig.add_subplot(gs[:, 2])    # math box     — right column

for a in [ax_cone, ax_curve, ax_math]:
    a.set_facecolor('#f8f8f6')

ax_math.axis('off')

# Slider axis
ax_sl = fig.add_axes([0.08, 0.04, 0.60, 0.025])
slider = Slider(ax_sl, 'Δz (µm)', -15, 15, valinit=0, valstep=0.1,
                color=AMBER, track_color='#e0ddd6')
slider.label.set_fontsize(10)

# ── Static parts of cone diagram ─────────────────────────────────────────────
Z_RANGE = 15   # µm shown in cone

def draw_cone(dz_um):
    ax_cone.clear()
    ax_cone.set_facecolor('#f8f8f6')

    zs = np.linspace(-Z_RANGE, 0, 300)  # beam travels left → right, focus at z=0

    # cone half-widths (scaled for display, not real nm)
    hw_x = np.abs(AX * zs * 1e-6) * 1e9 / D0_X * 1.8 + 0.3   # arb display units
    hw_y = np.abs(AY * zs * 1e-6) * 1e9 / D0_Y * 1.8 + 0.3

    ax_cone.fill_between(zs, -hw_x, hw_x, alpha=0.18, color=BLUE)
    ax_cone.plot(zs, hw_x,  color=BLUE, lw=1.5, label=f'X cone  α_x={AX*1e3:.3f} mrad')
    ax_cone.plot(zs, -hw_x, color=BLUE, lw=1.5)
    ax_cone.fill_between(zs, -hw_y, hw_y, alpha=0.25, color=TEAL)
    ax_cone.plot(zs, hw_y,  color=TEAL, lw=1.2, label=f'Y cone  α_y={AY*1e3:.3f} mrad')
    ax_cone.plot(zs, -hw_y, color=TEAL, lw=1.2)

    # Optical axis
    ax_cone.axhline(0, color='#aaa', lw=0.7, ls='--', zorder=0)

    # Focus point
    ax_cone.plot(0, 0, 'o', color=RED, ms=7, zorder=5)
    ax_cone.text(0.4, 0.22, 'focus  z=0', color=RED, fontsize=8, va='bottom')

    # DoF_x band
    ax_cone.axvspan(-DOF_X*1e6, 0, alpha=0.06, color=BLUE)
    ax_cone.text(-DOF_X*1e6/2, -2.8, f'DoF_x\n±{DOF_X*1e6:.2f} µm',
                 color=BLUE, fontsize=7.5, ha='center', va='top')

    # Sample plane
    r = calc(dz_um)
    sx = dz_um
    col = '#2ecc71' if (r['ok_x'] and r['ok_y']) else RED if not r['ok_x'] else AMBER
    ax_cone.axvline(sx, color=col, lw=2.5, ls='-', label='sample position', zorder=4)

    # Spot ellipse at sample
    spot_hw_x = r['dx_nm'] / D0_X * 1.8 + 0.3
    spot_hw_y = r['dy_nm'] / D0_Y * 1.8 + 0.3
    theta = np.linspace(0, 2*np.pi, 80)
    ax_cone.fill(sx + 0.25*np.cos(theta), spot_hw_x*np.sin(theta),
                 alpha=0.25, color=RED)
    ax_cone.plot(sx + 0.25*np.cos(theta), spot_hw_x*np.sin(theta),
                 color=RED, lw=1.2)

    # Angle annotation lines from focus to sample
    if abs(dz_um) > 0.3:
        hw_at_s_x = abs(AX * dz_um * 1e-6) * 1e9 / D0_X * 1.8
        ax_cone.annotate('', xy=(sx, hw_at_s_x), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color=BLUE, lw=1))
        ax_cone.annotate('', xy=(sx, -hw_at_s_x), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color=BLUE, lw=1))
        mid_z = sx / 2
        ax_cone.text(mid_z, hw_at_s_x * 0.55,
                     f'α_x={AX*1e3:.2f} mrad', color=BLUE,
                     fontsize=8, ha='center', va='bottom',
                     bbox=dict(fc='#f8f8f6', ec='none', pad=1))

    # Labels
    ax_cone.set_xlim(-Z_RANGE - 0.5, 2.5)
    ax_cone.set_ylim(-3.2, 3.2)
    ax_cone.set_xlabel('z position relative to focus (µm)', fontsize=9)
    ax_cone.set_ylabel('beam width (scaled)', fontsize=9)
    ax_cone.legend(fontsize=8, loc='upper left', framealpha=0.6)
    ax_cone.set_title(f'Convergence cone   —   sample at Δz = {dz_um:+.2f} µm', fontsize=10)
    ax_cone.tick_params(labelsize=8)
    ax_cone.grid(True, lw=0.3, alpha=0.4)


# ── d(Δz) curve ──────────────────────────────────────────────────────────────
zs_curve = np.linspace(-Z_RANGE, Z_RANGE, 400)
dx_curve = np.array([calc(z)['dx_nm'] for z in zs_curve])
dy_curve = np.array([calc(z)['dy_nm'] for z in zs_curve])

ln_x, = ax_curve.plot(zs_curve, dx_curve, color=BLUE, lw=2.5, label='d_x(Δz)')
ln_y, = ax_curve.plot(zs_curve, dy_curve, color=TEAL, lw=2.5, label='d_y(Δz)')
ax_curve.axhline(D0_X*1e9, color=BLUE, lw=0.8, ls='--', alpha=0.5)
ax_curve.axhline(D0_Y*1e9, color=TEAL, lw=0.8, ls='--', alpha=0.5)
ax_curve.axvspan(-DOF_X*1e6, DOF_X*1e6, alpha=0.07, color=BLUE,
                  label=f'DoF_x = ±{DOF_X*1e6:.2f} µm')
ax_curve.axvline(0, color=RED, lw=0.7, ls=':', alpha=0.5)
ax_curve.set_xlabel('Δz from focus (µm)', fontsize=9)
ax_curve.set_ylabel('beam size (nm)', fontsize=9)
ax_curve.legend(fontsize=8, ncol=4, loc='upper center')
ax_curve.grid(True, lw=0.3, alpha=0.4)
ax_curve.set_facecolor('#f8f8f6')
ax_curve.tick_params(labelsize=8)

# Moving cursor elements
vline_curve = ax_curve.axvline(0, color=AMBER, lw=2)
dot_x = ax_curve.plot(0, D0_X*1e9, 'o', color=BLUE, ms=8, zorder=5)[0]
dot_y = ax_curve.plot(0, D0_Y*1e9, 'o', color=TEAL, ms=8, zorder=5)[0]

# Math text object
math_obj = ax_math.text(
    0.05, 0.98, math_text(calc(0)),
    transform=ax_math.transAxes,
    fontsize=9, va='top', ha='left',
    fontfamily='monospace',
    bbox=dict(fc='white', ec='#cccccc', lw=0.8, pad=8, boxstyle='round,pad=0.5')
)

fig.suptitle('PBW beam convergence — interactive focal plane explorer',
             fontsize=12, fontweight='normal', y=0.97)


# ── Update function ───────────────────────────────────────────────────────────
def update(val):
    dz_um = slider.val
    r = calc(dz_um)

    # Cone
    draw_cone(dz_um)

    # Curve cursor
    vline_curve.set_xdata([dz_um, dz_um])
    dot_x.set_data([dz_um], [r['dx_nm']])
    dot_y.set_data([dz_um], [r['dy_nm']])

    # Math box
    math_obj.set_text(math_text(r))

    fig.canvas.draw_idle()


slider.on_changed(update)
draw_cone(0)   # initial draw

plt.savefig('beam_interactive_preview.png', dpi=130, bbox_inches='tight')
plt.show()
