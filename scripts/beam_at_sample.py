"""
Beam convergence at sample — CIBA Oxford Triplet
=================================================
Given a z-offset from focus, shows the live math for:
  Step 1: α_img = α_obj × D
  Step 2: cone blur = 2α × |Δz|
  Step 3: d(Δz) = √(d₀² + (2α·Δz)²)
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Fixed system values (from Table 2-2 and §2.6) ──────────────────────────
D0_X   = 9.3e-9          # spot at focus, X [m]
D0_Y   = 32e-9           # spot at focus, Y [m]
DX     = 857             # demagnification X
DY     = 130             # demagnification Y
A_OBJ  = 3e-6            # object divergence half-angle [rad]  §2.6 text

# ── Derived (Step 1) ────────────────────────────────────────────────────────
AX = A_OBJ * DX          # convergence half-angle X [rad]
AY = A_OBJ * DY          # convergence half-angle Y [rad]
DOF_X = D0_X / (2 * AX)  # depth of focus X [m]
DOF_Y = D0_Y / (2 * AY)  # depth of focus Y [m]


def math_at_z(dz_um: float) -> dict:
    """
    Show the three-step math for a given z-offset in µm.
    Returns a dict of every intermediate and final value.
    """
    dz = dz_um * 1e-6                        # convert to metres

    cone_x = 2 * AX * abs(dz)               # Step 2: cone blur X [m]
    cone_y = 2 * AY * abs(dz)               # Step 2: cone blur Y [m]

    d_x = np.sqrt(D0_X**2 + cone_x**2)      # Step 3: total spot X [m]
    d_y = np.sqrt(D0_Y**2 + cone_y**2)      # Step 3: total spot Y [m]

    return dict(
        dz_um   = dz_um,
        ax_mrad = AX * 1e3,
        ay_mrad = AY * 1e3,
        cone_x_nm = cone_x * 1e9,
        cone_y_nm = cone_y * 1e9,
        d_x_nm  = d_x * 1e9,
        d_y_nm  = d_y * 1e9,
        x_growth_pct = (d_x / D0_X - 1) * 100,
        y_growth_pct = (d_y / D0_Y - 1) * 100,
        inside_dof_x = abs(dz) <= DOF_X,
        inside_dof_y = abs(dz) <= DOF_Y,
    )


def print_math(dz_um: float):
    r = math_at_z(dz_um)
    print(f"\n{'─'*55}")
    print(f"  Δz = {dz_um:+.1f} µm from focus")
    print(f"{'─'*55}")
    print(f"  Step 1 — convergence angles (α_obj × D)")
    print(f"    α_x = {A_OBJ*1e6:.1f} µrad × {DX} = {r['ax_mrad']:.3f} mrad")
    print(f"    α_y = {A_OBJ*1e6:.1f} µrad × {DY} = {r['ay_mrad']:.3f} mrad")
    print(f"\n  Step 2 — cone blur = 2α × |Δz|")
    print(f"    2α_x × |Δz| = 2 × {r['ax_mrad']:.3f} × {abs(dz_um):.1f}×10⁻⁶ m = {r['cone_x_nm']:.2f} nm")
    print(f"    2α_y × |Δz| = 2 × {r['ay_mrad']:.3f} × {abs(dz_um):.1f}×10⁻⁶ m = {r['cone_y_nm']:.2f} nm")
    print(f"\n  Step 3 — d(Δz) = √(d₀² + (2α·Δz)²)")
    print(f"    d_x = √({D0_X*1e9:.1f}² + {r['cone_x_nm']:.2f}²) = {r['d_x_nm']:.2f} nm  [{r['x_growth_pct']:+.1f}%]")
    print(f"    d_y = √({D0_Y*1e9:.1f}² + {r['cone_y_nm']:.2f}²) = {r['d_y_nm']:.2f} nm  [{r['y_growth_pct']:+.1f}%]")
    print(f"\n  Inside DoF_x ({DOF_X*1e6:.2f} µm)?  {'YES ✓' if r['inside_dof_x'] else 'NO  ✗'}")
    print(f"  Inside DoF_y ({DOF_Y*1e6:.1f} µm)?  {'YES ✓' if r['inside_dof_y'] else 'NO  ✗'}")


def plot(z_range_um=15):
    zs = np.linspace(-z_range_um, z_range_um, 400)
    dx = [math_at_z(z)['d_x_nm'] for z in zs]
    dy = [math_at_z(z)['d_y_nm'] for z in zs]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(zs, dx, color='#185FA5', lw=2.5, label='d_x(Δz)')
    ax.plot(zs, dy, color='#0F6E56', lw=2.5, label='d_y(Δz)')
    ax.axhline(D0_X*1e9, color='#185FA5', lw=0.8, ls='--', alpha=0.5, label=f'd₀_x = {D0_X*1e9:.1f} nm')
    ax.axhline(D0_Y*1e9, color='#0F6E56', lw=0.8, ls='--', alpha=0.5, label=f'd₀_y = {D0_Y*1e9:.1f} nm')
    ax.axvspan(-DOF_X*1e6, DOF_X*1e6, alpha=0.07, color='#185FA5', label=f'DoF_x = ±{DOF_X*1e6:.2f} µm')
    ax.axvline(0, color='#A32D2D', lw=0.8, ls=':', alpha=0.6, label='focus')
    ax.set_xlabel('Δz from focus (µm)')
    ax.set_ylabel('beam size (nm)')
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, lw=0.4, alpha=0.4)
    plt.tight_layout()
    plt.savefig('beam_at_sample.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for z in [0, 1, 2, 5, -3, 10]:
        print_math(z)
    plot()
