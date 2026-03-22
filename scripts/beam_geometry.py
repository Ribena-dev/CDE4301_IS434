"""
PBW geometric convergence explorer
===================================
Three linked views update as you drag the Δz slider:
  - Side view:  cone cross-section with sample plane
  - Front view: beam ellipse at sample surface
  - Curve:      d(Δz) hyperbola with current position

The math box shows exactly how α_x is read from the geometry
at any sample position.

Run:  python beam_geometry.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from matplotlib.patches import Arc, FancyArrow

# ── Constants (Table 2-2 and §2.6) ──────────────────────────────────────────
D0X, D0Y = 9.3e-9, 32e-9        # spot at focus [m]
AX  = 3e-6 * 857                 # convergence angle X [rad]  = α_obj × Dx
AY  = 3e-6 * 130                 # convergence angle Y [rad]
DOFX = D0X / (2 * AX)           # depth of focus X [m]
DOFY = D0Y / (2 * AY)           # depth of focus Y [m]

B, T, A, R, GR = '#185FA5','#0F6E56','#BA7517','#A32D2D','#888780'


def spot(dz_um):
    dz = dz_um * 1e-6
    cbx = 2 * AX * abs(dz) * 1e9   # nm
    cby = 2 * AY * abs(dz) * 1e9
    dx  = np.sqrt(D0X**2 + (2*AX*abs(dz))**2) * 1e9
    dy  = np.sqrt(D0Y**2 + (2*AY*abs(dz))**2) * 1e9
    return cbx, cby, dx, dy


# ── Layout ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8), facecolor='#f8f8f6')
gs  = gridspec.GridSpec(2, 3, figure=fig,
                        left=0.06, right=0.97,
                        top=0.90,  bottom=0.14,
                        hspace=0.42, wspace=0.35)

ax_side  = fig.add_subplot(gs[0, :2])   # side cone view
ax_front = fig.add_subplot(gs[1, 0])    # beam cross-section
# ax_curve = fig.add_subplot(gs[1, 1])    # d(Δz) curve
ax_math  = fig.add_subplot(gs[:, 2])    # math box

for ax in [ax_side, ax_front, ax_math]:
    ax.set_facecolor('#f8f8f6')
ax_math.axis('off')

ax_sl = fig.add_axes([0.08, 0.04, 0.58, 0.025])
slider = Slider(ax_sl, 'Sample Δz (µm)', -20, 20,
                valinit=-8, valstep=0.1, color=A, track_color='#e0ddd6')
slider.label.set_fontsize(10)


# ── Static d(Δz) curve ───────────────────────────────────────────────────────
Z = np.linspace(-20, 20, 600)
DX_curve = np.array([spot(z)[2] for z in Z])
DY_curve = np.array([spot(z)[3] for z in Z])

# ax_curve.plot(Z, DX_curve, color=B, lw=2.5, label='d_x (hyperbola)')
# ax_curve.plot(Z, DY_curve, color=T, lw=2.5, label='d_y')
# # linear asymptote
# ax_curve.plot(Z, np.abs(Z)*2*AX*1e9 + D0X*1e9*0,
#               color=GR, lw=1, ls='--', alpha=0.55, label='2α_x·|Δz| (asymptote)')
# ax_curve.axhline(D0X*1e9, color=B, lw=0.8, ls=':', alpha=0.5)
# ax_curve.axhline(D0Y*1e9, color=T, lw=0.8, ls=':', alpha=0.5)
# ax_curve.axhline(np.sqrt(2)*D0X*1e9, color=R, lw=0.7, ls=':', alpha=0.4)
# ax_curve.axvspan(-DOFX*1e6, DOFX*1e6, alpha=0.07, color=B,
#                  label=f'DoF_x = ±{DOFX*1e6:.2f} µm')
# ax_curve.axvline(0, color=R, lw=0.7, ls=':', alpha=0.4)
# ax_curve.set_xlabel('Δz from focus (µm)', fontsize=9)
# ax_curve.set_ylabel('beam size (nm)', fontsize=9)
# ax_curve.legend(fontsize=7.5, loc='upper center', ncol=2)
# ax_curve.grid(True, lw=0.3, alpha=0.35)
# ax_curve.tick_params(labelsize=8)
# ax_curve.set_title('d(Δz) = √(d₀² + (2α·Δz)²)', fontsize=9)

# # Moving elements on curve
# cur_vline = ax_curve.axvline(0, color=A, lw=2)
# cur_dot_x = ax_curve.plot([], [], 'o', color=B, ms=8, zorder=5)[0]
# cur_dot_y = ax_curve.plot([], [], 'o', color=T, ms=8, zorder=5)[0]


# ── Draw side view ────────────────────────────────────────────────────────────
def draw_side(dz_um):
    ax_side.clear()
    ax_side.set_facecolor('#f8f8f6')

    # Geometry: focus at z=0, beam arrives from left.
    # Display: x-axis = Δz in µm, y-axis = half-width in scaled units.
    # Cone half-width at position z (µm): hw_x = |AX * z * 1e-6| * 1e9 (nm), scaled for display.
    SCALE = 0.6   # nm → display units

    z_range = np.linspace(-20, 4, 400)

    # Full cone walls (extend slightly past focus to show diverging side)
    z_full = np.linspace(-20, 4, 400)
    hw_x = np.abs(AX * z_full * 1e-6) * 1e9 * SCALE
    hw_y = np.abs(AY * z_full * 1e-6) * 1e9 * SCALE

    ax_side.fill_between(z_full, -hw_x, hw_x, alpha=0.12, color=B)
    ax_side.plot(z_full,  hw_x, color=B, lw=1.8, label=f'X cone  α_x = {AX*1e3:.3f} mrad')
    ax_side.plot(z_full, -hw_x, color=B, lw=1.8)
    ax_side.fill_between(z_full, -hw_y, hw_y, alpha=0.20, color=T)
    ax_side.plot(z_full,  hw_y, color=T, lw=1.2, label=f'Y cone  α_y = {AY*1e3:.3f} mrad')
    ax_side.plot(z_full, -hw_y, color=T, lw=1.2)

    # Optical axis
    ax_side.axhline(0, color='#aaa', lw=0.7, ls='--', zorder=0)

    # DoF_x band
    ax_side.axvspan(-DOFX*1e6, 0, alpha=0.07, color=B)
    ax_side.axvspan(0, DOFX*1e6, alpha=0.07, color=B)
    ax_side.text(-DOFX*1e6*0.5, -13, f'DoF_x\n±{DOFX*1e6:.2f}µm',
                 color=B, fontsize=7.5, ha='center', va='top')

    # Focus
    ax_side.plot(0, 0, 'o', color=R, ms=8, zorder=5)
    ax_side.text(0.3, 1.0, 'z=0 (focus)', color=R, fontsize=8.5, va='bottom')

    # ── Sample plane ──
    sx = dz_um
    cbx, cby, dx_nm, dy_nm = spot(dz_um)
    ok = abs(dz_um) <= DOFX*1e6
    col = '#2ecc71' if ok else R

    ax_side.axvline(sx, color=col, lw=2.8, label=f'sample  Δz = {dz_um:+.1f} µm')

    # Beam footprint bar at sample
    hw_at_s = abs(AX * dz_um * 1e-6) * 1e9 * SCALE + D0X*1e9*SCALE*0.1
    ax_side.plot([sx, sx], [-hw_at_s, hw_at_s], color=R, lw=4, alpha=0.5, solid_capstyle='round')

    # ── Key geometry: α_x angle lines from focus to sample ──
    if abs(dz_um) > 0.3:
        hw = abs(AX * dz_um * 1e-6) * 1e9 * SCALE
        ax_side.annotate('', xy=(sx, hw), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color=B, lw=1.3))
        ax_side.annotate('', xy=(sx, -hw), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color=B, lw=1.3))

        # α_x angle arc at focus
        arc = Arc((0, 0), 8, 8,
                  angle=0,
                  theta1=np.degrees(np.arctan2(hw, abs(sx))) * (-1 if sx < 0 else 1) * (-1),
                  theta2=np.degrees(np.arctan2(hw, abs(sx))) * (1 if sx < 0 else -1) * (-1),
                  color=B, lw=1.5)
        # simpler: just draw the arc manually
        th = np.linspace(0, np.arctan2(hw, abs(sx)), 30)
        r_arc = 5
        if sx < 0:
            ax_side.plot(r_arc * np.cos(np.pi - th),  r_arc * np.sin(th), color=B, lw=1.5)
            ax_side.plot(r_arc * np.cos(np.pi - th), -r_arc * np.sin(th), color=B, lw=1.5)
            ax_side.text(-r_arc - 1, r_arc * 0.4,
                         f'α_x\n{AX*1e3:.2f}\nmrad', color=B, fontsize=8, ha='right', va='center')
        else:
            ax_side.plot(r_arc * np.cos(th),  r_arc * np.sin(th), color=B, lw=1.5)
            ax_side.plot(r_arc * np.cos(th), -r_arc * np.sin(th), color=B, lw=1.5)
            ax_side.text(r_arc + 1, r_arc * 0.4,
                         f'α_x\n{AX*1e3:.2f}\nmrad', color=B, fontsize=8, ha='left', va='center')

        # Cone-blur dimension line
        mid_z = sx * 0.55
        ax_side.annotate('', xy=(sx, hw), xytext=(sx, 0),
                         arrowprops=dict(arrowstyle='<->', color=A, lw=1.2))
        ax_side.text(sx + 0.4, hw * 0.5, f'α_x·|Δz|\n= {cbx/2:.1f} nm',
                     color=A, fontsize=7.5, va='center')

    # Δz dimension arrow along axis
    if abs(dz_um) > 0.5:
        ax_side.annotate('', xy=(0, -15), xytext=(sx, -15),
                         arrowprops=dict(arrowstyle='<->', color=col, lw=1.2))
        ax_side.text(sx / 2, -17, f'|Δz| = {abs(dz_um):.1f} µm',
                     color=col, fontsize=8, ha='center', va='top')

    ax_side.set_xlim(-21, 5)
    ax_side.set_ylim(-20, 20)
    ax_side.set_xlabel('z relative to focus (µm)', fontsize=9)
    ax_side.set_ylabel('beam half-width (scaled nm)', fontsize=9)
    ax_side.legend(fontsize=8, loc='upper right', framealpha=0.6)
    ax_side.set_title(f'Side view — cone geometry   Δz = {dz_um:+.2f} µm', fontsize=10)
    ax_side.grid(True, lw=0.3, alpha=0.35)
    ax_side.tick_params(labelsize=8)


# ── Draw front view ───────────────────────────────────────────────────────────
def draw_front(dz_um):
    ax_front.clear()
    ax_front.set_facecolor('#f8f8f6')

    cbx, cby, dx_nm, dy_nm = spot(dz_um)
    theta = np.linspace(0, 2*np.pi, 200)

    # d₀ reference ellipse (dashed)
    ax_front.plot(D0X*1e9*np.cos(theta), D0Y*1e9*np.sin(theta),
                  color=GR, lw=1, ls='--', alpha=0.6, label='d₀ at focus')

    # Actual ellipse at Δz
    ax_front.fill(dx_nm*np.cos(theta), dy_nm*np.sin(theta),
                  alpha=0.18, color=R)
    ax_front.plot(dx_nm*np.cos(theta), dy_nm*np.sin(theta),
                  color=R, lw=2, label=f'd at Δz={dz_um:+.1f}µm')

    # Crosshairs
    ax_front.axhline(0, color='#aaa', lw=0.5, ls='--', alpha=0.4)
    ax_front.axvline(0, color='#aaa', lw=0.5, ls='--', alpha=0.4)

    # Dimension arrows
    ax_front.annotate('', xy=(dx_nm, 0), xytext=(-dx_nm, 0),
                      arrowprops=dict(arrowstyle='<->', color=B, lw=1.2))
    ax_front.text(0, -dy_nm*0.35, f'd_x = {dx_nm:.1f} nm',
                  color=B, ha='center', fontsize=8.5, fontweight='bold')

    ax_front.annotate('', xy=(0, dy_nm), xytext=(0, -dy_nm),
                      arrowprops=dict(arrowstyle='<->', color=T, lw=1.2))
    ax_front.text(dx_nm*0.55, 0, f'd_y =\n{dy_nm:.1f} nm',
                  color=T, ha='left', fontsize=8.5, fontweight='bold', va='center')

    lim = max(dx_nm, dy_nm) * 1.5
    ax_front.set_xlim(-lim, lim)
    ax_front.set_ylim(-lim, lim)
    ax_front.set_aspect('equal')
    ax_front.set_xlabel('X (nm)', fontsize=9)
    ax_front.set_ylabel('Y (nm)', fontsize=9)
    ax_front.set_title('Beam cross-section at sample', fontsize=9)
    ax_front.legend(fontsize=7.5, loc='upper right')
    ax_front.tick_params(labelsize=7)


# ── Math box ──────────────────────────────────────────────────────────────────
def make_math(dz_um):
    cbx, cby, dx_nm, dy_nm = spot(dz_um)
    dz = abs(dz_um) * 1e-6
    # Derive α_x from geometry:
    # tan(α_x) = cone_half_width / |Δz|
    # cone_half_width = cbx/2 nm = cbx/2 e-9 m
    if abs(dz_um) > 0.05:
        a_geo = (cbx/2 * 1e-9) / dz  # rad
    else:
        a_geo = AX
    ok_x = abs(dz_um) <= DOFX*1e6
    ok_y = abs(dz_um) <= DOFY*1e6
    region = "before focus" if dz_um < 0 else "past focus" if dz_um > 0 else "AT FOCUS"

    return (
        f"┌─ Geometric reading of α_x ─────────────────┐\n\n"
        f"  Position: Δz = {dz_um:+.2f} µm  ({region})\n\n"
        f"  The cone wall makes angle α_x with the\n"
        f"  optical axis. From the right triangle:\n\n"
        f"    tan(α_x) = cone half-width / |Δz|\n"
        f"             = ({cbx/2:.3f} nm) / ({abs(dz_um):.2f} µm)\n"
        f"             = {a_geo*1e3:.4f} mrad\n"
        f"             ≈ α_x = {AX*1e3:.3f} mrad  ✓\n\n"
        f"  (tan(α) ≈ α for small angles in radians)\n\n"
        f"─────────────────────────────────────────────\n\n"
        f"  Cone blur   2α_x·|Δz| = {cbx:.3f} nm\n"
        f"  Cone blur   2α_y·|Δz| = {cby:.3f} nm\n\n"
        f"  d_x = √({D0X*1e9:.1f}² + {cbx:.2f}²)\n"
        f"      = {dx_nm:.2f} nm  [{(dx_nm/D0X/1e9-1)*100:+.1f}%]\n\n"
        f"  d_y = √({D0Y*1e9:.1f}² + {cby:.2f}²)\n"
        f"      = {dy_nm:.2f} nm  [{(dy_nm/D0Y/1e9-1)*100:+.1f}%]\n\n"
        f"─────────────────────────────────────────────\n\n"
        f"  DoF_x = {DOFX*1e6:.2f} µm  →  {'INSIDE ✓' if ok_x else 'OUTSIDE ✗'}\n"
        f"  DoF_y = {DOFY*1e6:.1f} µm  →  {'INSIDE ✓' if ok_y else 'OUTSIDE ✗'}\n\n"
        f"└─────────────────────────────────────────────┘"
    )


math_txt = ax_math.text(
    0.04, 0.98, make_math(-8),
    transform=ax_math.transAxes,
    fontsize=8.5, va='top', ha='left',
    fontfamily='monospace',
    bbox=dict(fc='white', ec='#cccccc', lw=0.8, pad=8, boxstyle='round,pad=0.5')
)

fig.suptitle('PBW beam geometry — drag slider to find α_x at any focal plane position',
             fontsize=11, y=0.97)


# ── Update ────────────────────────────────────────────────────────────────────
def update(val):
    dz = slider.val
    cbx, cby, dx_nm, dy_nm = spot(dz)

    draw_side(dz)
    draw_front(dz)

    # cur_vline.set_xdata([dz, dz])
    # cur_dot_x.set_data([dz], [dx_nm])
    # cur_dot_y.set_data([dz], [dy_nm])

    math_txt.set_text(make_math(dz))
    fig.canvas.draw_idle()


slider.on_changed(update)
draw_side(-8)
draw_front(-8)
# cur_dot_x.set_data([-8], [spot(-8)[2]])
# cur_dot_y.set_data([-8], [spot(-8)[3]])
# cur_vline.set_xdata([-8, -8])

plt.savefig('beam_geometry.png', dpi=130, bbox_inches='tight')
plt.show()
