"""
PBW geometric convergence explorer  (v2 — aperture/slit controls)
===================================================================
Four linked sliders:
  - Δz        : sample position relative to focus
  - X slit    : horizontal slit half-opening (µm) → controls α_x and d₀_x
  - Y slit    : vertical   slit half-opening (µm) → controls α_y and d₀_y

Physics model (emittance-conserved):
  α   ∝  slit opening              (larger slit → wider cone → tighter focus)
  d₀  =  emittance / α             (emittance = d₀_ref × α_ref = constant per axis)
  DoF =  d₀ / (2α)  =  emit / (2α²)   (shrinks as aperture opens)

Three linked views update on any slider change:
  - Side view  : cone cross-section with sample plane
  - Front view : beam ellipse at sample surface
  - Math box   : live derivation of all parameters

Run:  python beam_geometry.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

# ── Reference constants (Table 2-2, §2.6) ───────────────────────────────────
# These are the CIBA baseline values at the reference slit opening.
D0X_REF = 9.3e-9        # spot at focus, X axis [m]
D0Y_REF = 32e-9         # spot at focus, Y axis [m]
AX_REF  = 3e-6 * 857    # convergence angle X [rad]
AY_REF  = 3e-6 * 130    # convergence angle Y [rad]

SLIT_X_REF = 100.0      # reference X slit half-opening [µm]
SLIT_Y_REF = 100.0      # reference Y slit half-opening [µm]

# Emittance (conserved per axis):  ε = d₀ · α  [m·rad]
EMIT_X = D0X_REF * AX_REF
EMIT_Y = D0Y_REF * AY_REF

# ── Colour palette ───────────────────────────────────────────────────────────
B, T, A, R, GR = '#185FA5', '#0F6E56', '#BA7517', '#A32D2D', '#888780'
BG = '#f8f8f6'


# ── Physics: derive beam parameters from slit opening ───────────────────────
def beam_params(slit_x_um, slit_y_um):
    """Return (ax, ay, d0x, d0y, dofx, dofy) for given slit half-openings."""
    ax   = AX_REF  * (slit_x_um / SLIT_X_REF)
    ay   = AY_REF  * (slit_y_um / SLIT_Y_REF)
    d0x  = EMIT_X  / ax          # emittance conservation
    d0y  = EMIT_Y  / ay
    dofx = d0x / (2 * ax)
    dofy = d0y / (2 * ay)
    return ax, ay, d0x, d0y, dofx, dofy


def spot(dz_um, ax, ay, d0x, d0y):
    """Beam size at sample position Δz from focus."""
    dz  = dz_um * 1e-6
    cbx = 2 * ax * abs(dz) * 1e9       # cone blur X [nm]
    cby = 2 * ay * abs(dz) * 1e9       # cone blur Y [nm]
    dx  = np.sqrt(d0x**2 + (2*ax*abs(dz))**2) * 1e9   # total X [nm]
    dy  = np.sqrt(d0y**2 + (2*ay*abs(dz))**2) * 1e9   # total Y [nm]
    return cbx, cby, dx, dy


# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig,
                        left=0.06, right=0.97,
                        top=0.90,  bottom=0.22,    # extra bottom space for 3 sliders
                        hspace=0.42, wspace=0.35)

ax_side  = fig.add_subplot(gs[0, :2])
ax_front = fig.add_subplot(gs[1, 0])
ax_math  = fig.add_subplot(gs[:, 2])

for ax in [ax_side, ax_front, ax_math]:
    ax.set_facecolor(BG)
ax_math.axis('off')

# ── Sliders ───────────────────────────────────────────────────────────────────
# Row positions (bottom-up)
SL_DZ   = [0.05, 0.025]   # [y_start, height]
SL_SX   = [0.11, 0.025]
SL_SY   = [0.17, 0.025]

ax_sl_dz = fig.add_axes([0.08, SL_DZ[0], 0.58, SL_DZ[1]])
ax_sl_sx = fig.add_axes([0.08, SL_SX[0], 0.58, SL_SX[1]])
ax_sl_sy = fig.add_axes([0.08, SL_SY[0], 0.58, SL_SY[1]])

sl_dz = Slider(ax_sl_dz, 'Sample Δz (µm)',   -20,  20,  valinit=-8,   valstep=0.1,  color=A,  track_color='#e0ddd6')
sl_sx = Slider(ax_sl_sx, 'X slit (µm)',        10, 300, valinit=100,  valstep=1.0,  color=B,  track_color='#dde9f5')
sl_sy = Slider(ax_sl_sy, 'Y slit (µm)',        10, 300, valinit=100,  valstep=1.0,  color=T,  track_color='#d9f0e8')

for sl in [sl_dz, sl_sx, sl_sy]:
    sl.label.set_fontsize(9)
    sl.valtext.set_fontsize(9)


# ── Draw side view ────────────────────────────────────────────────────────────
def draw_side(dz_um, ax, ay, d0x, d0y, dofx, dofy):
    ax_side.clear()
    ax_side.set_facecolor(BG)

    SCALE = 0.6   # nm → display units
    z_full = np.linspace(-20, 4, 400)
    hw_x = np.abs(ax * z_full * 1e-6) * 1e9 * SCALE
    hw_y = np.abs(ay * z_full * 1e-6) * 1e9 * SCALE

    ax_side.fill_between(z_full, -hw_x, hw_x, alpha=0.12, color=B)
    ax_side.plot(z_full,  hw_x, color=B, lw=1.8, label=f'X cone  α_x = {ax*1e3:.3f} mrad')
    ax_side.plot(z_full, -hw_x, color=B, lw=1.8)
    ax_side.fill_between(z_full, -hw_y, hw_y, alpha=0.20, color=T)
    ax_side.plot(z_full,  hw_y, color=T, lw=1.2, label=f'Y cone  α_y = {ay*1e3:.3f} mrad')
    ax_side.plot(z_full, -hw_y, color=T, lw=1.2)

    ax_side.axhline(0, color='#aaa', lw=0.7, ls='--', zorder=0)

    # DoF_x band
    ax_side.axvspan(-dofx*1e6, 0,       alpha=0.07, color=B)
    ax_side.axvspan(0,          dofx*1e6, alpha=0.07, color=B)
    ax_side.text(-dofx*1e6*0.5, -13,
                 f'DoF_x\n±{dofx*1e6:.2f}µm',
                 color=B, fontsize=7.5, ha='center', va='top')

    # Focus marker
    ax_side.plot(0, 0, 'o', color=R, ms=8, zorder=5)
    ax_side.text(0.3, 1.0, 'z=0 (focus)', color=R, fontsize=8.5, va='bottom')

    # Sample plane
    sx  = dz_um
    cbx, cby, dx_nm, dy_nm = spot(dz_um, ax, ay, d0x, d0y)
    ok  = abs(dz_um) <= dofx*1e6
    col = '#2ecc71' if ok else R

    ax_side.axvline(sx, color=col, lw=2.8, label=f'sample  Δz = {dz_um:+.1f} µm')
    hw_at_s = abs(ax * dz_um * 1e-6) * 1e9 * SCALE + d0x*1e9*SCALE*0.1
    ax_side.plot([sx, sx], [-hw_at_s, hw_at_s], color=R, lw=4, alpha=0.5, solid_capstyle='round')

    # Angle lines and arc
    if abs(dz_um) > 0.3:
        hw = abs(ax * dz_um * 1e-6) * 1e9 * SCALE
        ax_side.annotate('', xy=(sx,  hw), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color=B, lw=1.3))
        ax_side.annotate('', xy=(sx, -hw), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color=B, lw=1.3))

        th    = np.linspace(0, np.arctan2(hw, abs(sx)), 30)
        r_arc = 5
        if sx < 0:
            ax_side.plot(r_arc * np.cos(np.pi - th),  r_arc * np.sin(th),  color=B, lw=1.5)
            ax_side.plot(r_arc * np.cos(np.pi - th), -r_arc * np.sin(th),  color=B, lw=1.5)
            ax_side.text(-r_arc - 1, r_arc * 0.4,
                         f'α_x\n{ax*1e3:.2f}\nmrad', color=B, fontsize=8, ha='right', va='center')
        else:
            ax_side.plot(r_arc * np.cos(th),  r_arc * np.sin(th),  color=B, lw=1.5)
            ax_side.plot(r_arc * np.cos(th), -r_arc * np.sin(th),  color=B, lw=1.5)
            ax_side.text(r_arc + 1, r_arc * 0.4,
                         f'α_x\n{ax*1e3:.2f}\nmrad', color=B, fontsize=8, ha='left', va='center')

        ax_side.annotate('', xy=(sx, hw), xytext=(sx, 0),
                         arrowprops=dict(arrowstyle='<->', color=A, lw=1.2))
        ax_side.text(sx + 0.4, hw * 0.5, f'α_x·|Δz|\n= {cbx/2:.1f} nm',
                     color=A, fontsize=7.5, va='center')

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
def draw_front(dz_um, ax, ay, d0x, d0y):
    ax_front.clear()
    ax_front.set_facecolor(BG)

    cbx, cby, dx_nm, dy_nm = spot(dz_um, ax, ay, d0x, d0y)
    theta = np.linspace(0, 2*np.pi, 200)

    # Reference ellipse at focus (dashed)
    ax_front.plot(d0x*1e9*np.cos(theta), d0y*1e9*np.sin(theta),
                  color=GR, lw=1, ls='--', alpha=0.6, label='d₀ at focus')

    # Actual ellipse at Δz
    ax_front.fill(dx_nm*np.cos(theta), dy_nm*np.sin(theta), alpha=0.18, color=R)
    ax_front.plot(dx_nm*np.cos(theta), dy_nm*np.sin(theta),
                  color=R, lw=2, label=f'd at Δz={dz_um:+.1f}µm')

    ax_front.axhline(0, color='#aaa', lw=0.5, ls='--', alpha=0.4)
    ax_front.axvline(0, color='#aaa', lw=0.5, ls='--', alpha=0.4)

    ax_front.annotate('', xy=(dx_nm, 0), xytext=(-dx_nm, 0),
                      arrowprops=dict(arrowstyle='<->', color=B, lw=1.2))
    ax_front.text(0, -dy_nm*0.35, f'd_x = {dx_nm:.1f} nm',
                  color=B, ha='center', fontsize=8.5, fontweight='bold')

    ax_front.annotate('', xy=(0, dy_nm), xytext=(0, -dy_nm),
                      arrowprops=dict(arrowstyle='<->', color=T, lw=1.2))
    ax_front.text(dx_nm*0.55, 0, f'd_y =\n{dy_nm:.1f} nm',
                  color=T, ha='left', fontsize=8.5, fontweight='bold', va='center')

    lim = max(dx_nm, dy_nm) * 1.6
    ax_front.set_xlim(-lim, lim)
    ax_front.set_ylim(-lim, lim)
    ax_front.set_aspect('equal')
    ax_front.set_xlabel('X (nm)', fontsize=9)
    ax_front.set_ylabel('Y (nm)', fontsize=9)
    ax_front.set_title('Beam cross-section at sample', fontsize=9)
    ax_front.legend(fontsize=7.5, loc='upper right')
    ax_front.tick_params(labelsize=7)


# ── Math box ──────────────────────────────────────────────────────────────────
def make_math(dz_um, slit_x, slit_y, ax, ay, d0x, d0y, dofx, dofy):
    cbx, cby, dx_nm, dy_nm = spot(dz_um, ax, ay, d0x, d0y)
    dz  = abs(dz_um) * 1e-6
    a_geo = (cbx/2 * 1e-9) / dz if abs(dz_um) > 0.05 else ax
    ok_x  = abs(dz_um) <= dofx*1e6
    ok_y  = abs(dz_um) <= dofy*1e6
    region = "before focus" if dz_um < 0 else "past focus" if dz_um > 0 else "AT FOCUS"

    return (
        f"┌─ Slit → beam parameters ───────────────────┐\n\n"
        f"  X slit = {slit_x:.0f} µm  (ref = {SLIT_X_REF:.0f} µm)\n"
        f"  Y slit = {slit_y:.0f} µm  (ref = {SLIT_Y_REF:.0f} µm)\n\n"
        f"  α_x = α_ref × (slit_x / slit_ref)\n"
        f"      = {AX_REF*1e3:.3f} × ({slit_x:.0f}/{SLIT_X_REF:.0f})\n"
        f"      = {ax*1e3:.3f} mrad\n\n"
        f"  d₀_x = ε_x / α_x  (emittance = {EMIT_X*1e18:.3f} nm·mrad)\n"
        f"       = {d0x*1e9:.2f} nm\n\n"
        f"  α_y = {ay*1e3:.3f} mrad   d₀_y = {d0y*1e9:.2f} nm\n\n"
        f"─────────────────────────────────────────────\n\n"
        f"  Position: Δz = {dz_um:+.2f} µm  ({region})\n\n"
        f"    tan(α_x) ≈ {a_geo*1e3:.4f} mrad  ✓\n\n"
        f"  Cone blur  2α_x·|Δz| = {cbx:.3f} nm\n"
        f"  Cone blur  2α_y·|Δz| = {cby:.3f} nm\n\n"
        f"  d_x = √(d₀_x² + blur_x²)\n"
        f"      = √({d0x*1e9:.2f}² + {cbx:.2f}²)\n"
        f"      = {dx_nm:.2f} nm  [{(dx_nm/(d0x*1e9)-1)*100:+.1f}% vs focus]\n\n"
        f"  d_y = {dy_nm:.2f} nm  [{(dy_nm/(d0y*1e9)-1)*100:+.1f}% vs focus]\n\n"
        f"─────────────────────────────────────────────\n\n"
        f"  DoF_x = {dofx*1e6:.3f} µm  → {'INSIDE ✓' if ok_x else 'OUTSIDE ✗'}\n"
        f"  DoF_y = {dofy*1e6:.3f} µm  → {'INSIDE ✓' if ok_y else 'OUTSIDE ✗'}\n\n"
        f"└─────────────────────────────────────────────┘"
    )


# ── Initialise text object ────────────────────────────────────────────────────
_p = beam_params(sl_sx.valinit, sl_sy.valinit)
math_txt = ax_math.text(
    0.04, 0.98,
    make_math(sl_dz.valinit, sl_sx.valinit, sl_sy.valinit, *_p),
    transform=ax_math.transAxes,
    fontsize=8.2, va='top', ha='left',
    fontfamily='monospace',
    bbox=dict(fc='white', ec='#cccccc', lw=0.8, pad=8, boxstyle='round,pad=0.5')
)

fig.suptitle(
    'PBW beam geometry — drag sliders to explore aperture, slit opening, and focal position',
    fontsize=11, y=0.97
)


# ── Update callback ───────────────────────────────────────────────────────────
def update(_val):
    dz     = sl_dz.val
    slit_x = sl_sx.val
    slit_y = sl_sy.val
    ax, ay, d0x, d0y, dofx, dofy = beam_params(slit_x, slit_y)

    draw_side(dz,  ax, ay, d0x, d0y, dofx, dofy)
    draw_front(dz, ax, ay, d0x, d0y)
    math_txt.set_text(make_math(dz, slit_x, slit_y, ax, ay, d0x, d0y, dofx, dofy))
    fig.canvas.draw_idle()


sl_dz.on_changed(update)
sl_sx.on_changed(update)
sl_sy.on_changed(update)

# Initial draw
update(None)

plt.savefig('beam_geometry.png', dpi=130, bbox_inches='tight')
plt.show()
