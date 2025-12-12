#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transient integrate-to-steady solver for 2D advection-diffusion with
constant electrode flux boundary conditions.

This is a FIXED + refactored version of the pasted script.

Key fixes vs the original paste:
- Recomputes y-grid and velocity profile consistently for each design case.
- Uses a physically meaningful Reynolds number for parallel-plate flow.
- Uses a steady-state check based on BOTH (i) field changes and (ii) a
  1D control-volume mass balance (inlet/outlet/flux), rather than comparing
  domain mass-change to an "expected" rate.
- Keeps base-case and optimal-case variables isolated to prevent
  accidentally mixing numbers in the final comparison.

Notes / scope:
- The model still solves each species independently with prescribed Faradaic
  wall fluxes. It does NOT include kinetic coupling or a transport-limited
  current (so very high j can still drive near-depletion unless you add a
  limiter).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Geometry:
    L: float  # m
    H: float  # m
    Nx: int
    Ny: int


@dataclass(frozen=True)
class Flow:
    v_areal: float  # m^2/s (volumetric flow per unit width)


@dataclass(frozen=True)
class Electrochem:
    j: float  # A/m^2
    n: int = 2
    F: float = 96485.0

    @property
    def Npp(self) -> float:
        """Molar flux from Faraday's law [mol/m^2/s]."""
        return self.j / (self.n * self.F)


@dataclass(frozen=True)
class Diffusivities:
    D_Cl2: float = 1.2e-9
    D_CO: float = 2.0e-9
    D_CO2: float = 2.0e-9

    @property
    def Dmax(self) -> float:
        return max(self.D_Cl2, self.D_CO, self.D_CO2)


def trapezoid_weights(N: int, dy: float) -> np.ndarray:
    w = np.ones(N) * dy
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def make_grid(geom: Geometry) -> tuple[float, float, np.ndarray, np.ndarray]:
    dx = geom.L / (geom.Nx - 1)
    dy = geom.H / (geom.Ny - 1)
    x = np.linspace(0.0, geom.L, geom.Nx)
    y = np.linspace(-geom.H / 2.0, geom.H / 2.0, geom.Ny)
    return dx, dy, x, y


def make_parabolic_vx(y: np.ndarray, geom: Geometry, flow: Flow) -> tuple[np.ndarray, float, float]:
    """Return vx(x,y) array (shape Nx×Ny) and v_avg/v_max.

    For fully developed laminar flow between parallel plates, the standard
    parabolic profile satisfies v_avg = v_areal/H and v_max = 1.5*v_avg.
    """
    v_avg = flow.v_areal / geom.H
    v_max = 1.5 * v_avg
    vx_y = v_max * (1.0 - (2.0 * y / geom.H) ** 2)
    vx_y[vx_y < 0.0] = 0.0
    vx = np.tile(vx_y, (geom.Nx, 1))
    return vx, v_avg, v_max


def reynolds_parallel_plates(flow: Flow, geom: Geometry, rho: float = 1000.0, mu: float = 1e-3) -> float:
    """Re = rho * v_avg * Dh / mu with Dh = 2H for wide parallel plates.

    v_avg = v_areal/H ⇒ Re = rho * (v_areal/H) * (2H) / mu = 2*rho*v_areal/mu.
    """
    v_avg = flow.v_areal / geom.H
    Dh = 2.0 * geom.H
    return rho * v_avg * Dh / mu


def choose_dt(dx: float, dy: float, vx: np.ndarray, Dmax: float, safety: float = 0.2) -> float:
    vx_max = float(np.max(vx))
    dt_adv = dx / (vx_max + 1e-30)
    dt_diff = min(dx * dx, dy * dy) / (4.0 * Dmax + 1e-30)
    dt = safety * min(dt_adv, dt_diff)
    if not np.isfinite(dt) or dt <= 0.0:
        return 1e-6
    return dt


def integrate_to_steady(
    *,
    D: float,
    C_inlet: float,
    flux_top: float,
    flux_bot: float,
    geom: Geometry,
    flow: Flow,
    vx: np.ndarray,
    dt: float,
    max_steps: int = 400_000,
    tol_change: float = 1e-10,
    tol_mb_rel: float = 5e-3,
    check_every: int = 400,
    verbose: bool = False,
) -> tuple[np.ndarray, int, dict[str, float]]:
    """Explicit transient integrate-to-steady.

    Flux convention:
    - flux_top / flux_bot are molar fluxes [mol/m^2/s] defined POSITIVE INTO THE FLUID.

    Steady-state detection:
    - Field-change criterion: max(|Cnew-C|) < tol_change
    - Control-volume mass balance residual criterion (relative):
        residual = (outflow - inflow) - (flux_top + flux_bot)*L
      |residual|/scale < tol_mb_rel
    """

    dx, dy, _, y = make_grid(geom)
    weights = trapezoid_weights(geom.Ny, dy)

    # Initialize
    C = np.full((geom.Nx, geom.Ny), C_inlet, dtype=float)
    C[0, :] = C_inlet

    t0 = time.time()

    # Requested source term (for reference only; may be infeasible for consumption)
    net_source = (flux_top + flux_bot) * geom.L  # mol/(m·s) per unit width

    # Avoid "y unused" warnings in some linters (also handy for future diagnostics)
    _ = y

    last_diag: dict[str, float] = {}

    for step in range(1, max_steps + 1):
        # ------------------------------------------------------------------
        # Interior update (vectorized)
        # ------------------------------------------------------------------
        Cnew = C.copy()

        # Upwind x-advection (vx >= 0 assumed; matches the original paste's assumption)
        adv = vx[1:-1, 1:-1] * (C[1:-1, 1:-1] - C[0:-2, 1:-1]) / dx

        # Diffusion
        diff_x = D * (C[2:, 1:-1] - 2.0 * C[1:-1, 1:-1] + C[0:-2, 1:-1]) / (dx * dx)
        diff_y = D * (C[1:-1, 2:] - 2.0 * C[1:-1, 1:-1] + C[1:-1, 0:-2]) / (dy * dy)

        Cnew[1:-1, 1:-1] = C[1:-1, 1:-1] + dt * (-adv + diff_x + diff_y)

        # ------------------------------------------------------------------
        # Boundary conditions
        # ------------------------------------------------------------------
        # Inlet Dirichlet
        Cnew[0, :] = C_inlet
        # Outlet Neumann (zero x-gradient)
        Cnew[-1, :] = Cnew[-2, :]

        # Top/bottom Neumann (flux) implemented via a second-order one-sided
        # derivative at the wall (grid includes the wall nodes):
        #
        # Bottom wall (y=-H/2): dC/dy = -flux_bot/D
        #   (-3*C0 + 4*C1 - C2) / (2dy) = -flux_bot/D
        #   => C0 = (4*C1 - C2 + 2dy*flux_bot/D) / 3
        #
        # Top wall (y=+H/2): dC/dy = +flux_top/D
        #   (3*C_N-1 - 4*C_N-2 + C_N-3) / (2dy) = flux_top/D
        #   => C_N-1 = (4*C_N-2 - C_N-3 + 2dy*flux_top/D) / 3
        #
        # IMPORTANT: for consumption (flux < 0), a fixed flux can demand more
        # CO2 than is locally available, which would force negative C near the
        # wall. Instead of letting the solver go negative and clipping (which
        # breaks mass balance), we LIMIT the magnitude of negative flux by the
        # local adjacent concentration (mass-transfer-limited boundary).
        flux_top_eff = np.full(geom.Nx, float(flux_top))
        flux_bot_eff = np.full(geom.Nx, float(flux_bot))

        if geom.Ny < 3:
            raise ValueError("Ny must be >= 3 for second-order Neumann BCs.")

        if flux_top < 0.0:
            # Ensure C_top_wall >= 0 => flux >= ((C_{N-3} - 4*C_{N-2}) * D) / (2dy)
            min_flux = (Cnew[:, -3] - 4.0 * Cnew[:, -2]) * D / (2.0 * (dy + 1e-30))
            flux_top_eff = np.maximum(flux_top_eff, min_flux)
        if flux_bot < 0.0:
            # Ensure C_bottom_wall >= 0 => flux >= ((C2 - 4*C1) * D) / (2dy)
            min_flux = (Cnew[:, 2] - 4.0 * Cnew[:, 1]) * D / (2.0 * (dy + 1e-30))
            flux_bot_eff = np.maximum(flux_bot_eff, min_flux)

        # Apply wall concentrations implied by the (possibly limited) fluxes
        Cnew[:, 0] = (4.0 * Cnew[:, 1] - Cnew[:, 2] + 2.0 * dy * flux_bot_eff / (D + 1e-30)) / 3.0
        Cnew[:, -1] = (4.0 * Cnew[:, -2] - Cnew[:, -3] + 2.0 * dy * flux_top_eff / (D + 1e-30)) / 3.0

        # No-flux behavior at inlet was already set by Dirichlet; re-apply inlet
        # after walls to keep it exact.
        Cnew[0, :] = C_inlet

        # Physical non-negativity (should rarely trigger now for consumption BCs)
        np.maximum(Cnew, 0.0, out=Cnew)

        if step % check_every == 0:
            change = float(np.max(np.abs(Cnew - C)))

            # Control-volume mass balance at steady state:
            # (total out through outlet plane) - (total in through inlet plane) = ∫(wall fluxes) dx
            #
            # IMPORTANT: With a Dirichlet inlet (fixed C at x=0), there can be
            # a significant *diffusive* influx at the inlet plane. For sinks
            # like CO2, this can materially change the balance, so we must
            # include convection + diffusion at x=0 and x=L.
            #
            # NOTE: if flux is limited (consumption), the *achieved* source term
            # is smaller in magnitude than the requested Faradaic flux. That's
            # the physically correct outcome (mass-transfer limitation).
            # Outlet plane (x=L): total flux in +x direction is vx*C - D*dC/dx.
            # With our outlet Neumann (C[-1]=C[-2]) the diffusive component is ~0.
            conv_out = float(np.sum(vx[-1, :] * Cnew[-1, :] * weights))
            diff_out = float(-D * np.sum((Cnew[-1, :] - Cnew[-2, :]) / dx * weights))
            outflow = conv_out + diff_out

            # Inlet plane (x=0): convective inflow uses imposed Dirichlet value,
            # but there can also be diffusive inflow from the imposed boundary.
            conv_in = float(np.sum(vx[0, :] * C_inlet * weights))
            diff_in = float(-D * np.sum((Cnew[1, :] - Cnew[0, :]) / dx * weights))
            inflow = conv_in + diff_in

            net_source_eff = float(np.sum((flux_top_eff + flux_bot_eff)) * dx)  # mol/(m·s)
            residual = (outflow - inflow) - net_source_eff
            scale = max(abs(outflow), abs(inflow), abs(net_source_eff), 1e-30)
            rel_resid = abs(residual) / scale

            last_diag = {
                "outflow": outflow,
                "inflow": inflow,
                "conv_in": conv_in,
                "diff_in": diff_in,
                "conv_out": conv_out,
                "diff_out": diff_out,
                "net_source": net_source,
                "net_source_eff": net_source_eff,
                "flux_top_req": float(flux_top),
                "flux_bot_req": float(flux_bot),
                "flux_top_eff_mean": float(np.mean(flux_top_eff)),
                "flux_bot_eff_mean": float(np.mean(flux_bot_eff)),
                "flux_top_limited_frac": float(np.mean(np.abs(flux_top_eff - float(flux_top)) > 0.0)),
                "flux_bot_limited_frac": float(np.mean(np.abs(flux_bot_eff - float(flux_bot)) > 0.0)),
                "mb_residual": residual,
                "mb_rel": rel_resid,
                "max_change": change,
                "dt": dt,
                "steps": float(step),
                "runtime_s": time.time() - t0,
            }

            if verbose and step % (check_every * 25) == 0:
                print(
                    f"step {step:7d}  maxΔ={change:.3e}  "
                    f"out={outflow:.3e}  in={inflow:.3e}  src={net_source:.3e}  "
                    f"relMB={rel_resid:.3e}"
                )

            if change < tol_change and rel_resid < tol_mb_rel:
                return Cnew, step, last_diag

        C = Cnew

    # If we didn't meet the convergence criteria, return the last computed diagnostics
    # (so callers can still see mass-balance quality and any flux-limiting).
    if not last_diag:
        # Extremely early exit / unexpected: compute a minimal diagnostic.
        conv_out = float(np.sum(vx[-1, :] * C[-1, :] * weights))
        diff_out = float(-D * np.sum((C[-1, :] - C[-2, :]) / dx * weights))
        outflow = conv_out + diff_out
        conv_in = float(np.sum(vx[0, :] * C_inlet * weights))
        diff_in = float(-D * np.sum((C[1, :] - C[0, :]) / dx * weights))
        inflow = conv_in + diff_in
        residual = (outflow - inflow) - net_source
        scale = max(abs(outflow), abs(inflow), abs(net_source), 1e-30)
        last_diag = {
            "outflow": outflow,
            "inflow": inflow,
            "conv_in": conv_in,
            "diff_in": diff_in,
            "conv_out": conv_out,
            "diff_out": diff_out,
            "net_source": net_source,
            "mb_residual": residual,
            "mb_rel": abs(residual) / scale,
            "max_change": float("nan"),
            "dt": dt,
            "steps": float(max_steps),
            "runtime_s": time.time() - t0,
        }
    return C, max_steps, last_diag


def outlet_metrics(*, C: np.ndarray, vx: np.ndarray, y: np.ndarray, dy: float) -> dict[str, float]:
    weights = trapezoid_weights(len(y), dy)
    C_out = C[-1, :]
    vx_out = vx[-1, :]

    total = float(np.sum(vx_out * C_out * weights))
    top = y > 0
    bot = y < 0

    top_flow = float(np.sum(vx_out[top] * C_out[top] * weights[top]))
    bot_flow = float(np.sum(vx_out[bot] * C_out[bot] * weights[bot]))

    return {
        "total": total,
        "top": top_flow,
        "bottom": bot_flow,
        "mean_out": float(np.mean(C_out)),
        "max": float(np.max(C)),
    }


def run_case(
    *,
    name: str,
    geom: Geometry,
    flow: Flow,
    elec: Electrochem,
    diff: Diffusivities,
    C0_Cl2: float,
    C0_CO: float,
    C0_CO2: float,
    verbose_solver: bool = False,
) -> dict[str, object]:
    dx, dy, x, y = make_grid(geom)
    vx, v_avg, v_max = make_parabolic_vx(y, geom, flow)
    dt = choose_dt(dx, dy, vx, diff.Dmax)
    Re = reynolds_parallel_plates(flow, geom)

    Npp = elec.Npp

    # Wall fluxes (mol/m^2/s), positive into fluid
    flux_top_cl2 = +Npp
    flux_bot_cl2 = 0.0

    flux_top_co = 0.0
    flux_bot_co = +Npp

    flux_top_co2 = 0.0
    flux_bot_co2 = -Npp

    print("\n" + "=" * 72)
    print(f"{name} CASE")
    print("=" * 72)
    print(f"L={geom.L:.3f} m, H={geom.H*1e3:.3f} mm, Nx={geom.Nx}, Ny={geom.Ny}")
    print(f"v_areal={flow.v_areal:.3e} m²/s, v_avg={v_avg:.3f} m/s, v_max={v_max:.3f} m/s")
    print(f"Re (parallel plates) = {Re:.1f}  (constraint: <1000)")
    print(f"j={elec.j:.1f} A/m²  =>  N''={Npp:.3e} mol/(m²·s)")
    print(f"dx={dx:.3e} m, dy={dy:.3e} m, dt={dt:.3e} s")

    # Solve
    C_Cl2, steps_cl2, diag_cl2 = integrate_to_steady(
        D=diff.D_Cl2,
        C_inlet=C0_Cl2,
        flux_top=flux_top_cl2,
        flux_bot=flux_bot_cl2,
        geom=geom,
        flow=flow,
        vx=vx,
        dt=dt,
        verbose=verbose_solver,
    )
    C_CO, steps_co, diag_co = integrate_to_steady(
        D=diff.D_CO,
        C_inlet=C0_CO,
        flux_top=flux_top_co,
        flux_bot=flux_bot_co,
        geom=geom,
        flow=flow,
        vx=vx,
        dt=dt,
        verbose=False,
    )
    C_CO2, steps_co2, diag_co2 = integrate_to_steady(
        D=diff.D_CO2,
        C_inlet=C0_CO2,
        flux_top=flux_top_co2,
        flux_bot=flux_bot_co2,
        geom=geom,
        flow=flow,
        vx=vx,
        dt=dt,
        verbose=False,
    )

    # Outlet metrics
    m_cl2 = outlet_metrics(C=C_Cl2, vx=vx, y=y, dy=dy)
    m_co = outlet_metrics(C=C_CO, vx=vx, y=y, dy=dy)
    m_co2 = outlet_metrics(C=C_CO2, vx=vx, y=y, dy=dy)

    eta_cl2 = (m_cl2["top"] / m_cl2["total"]) if m_cl2["total"] > 1e-30 else 0.0
    eta_co = (m_co["bottom"] / m_co["total"]) if m_co["total"] > 1e-30 else 0.0

    cl2_in_bottom_frac = (m_cl2["bottom"] / m_cl2["total"]) if m_cl2["total"] > 1e-30 else 0.0
    co_in_top_frac = (m_co["top"] / m_co["total"]) if m_co["total"] > 1e-30 else 0.0

    # CO2 conversion (flow-based)
    co2_inlet_flow = flow.v_areal * C0_CO2
    co2_outlet_flow = m_co2["total"]
    co2_conv = (co2_inlet_flow - co2_outlet_flow) / co2_inlet_flow * 100.0 if co2_inlet_flow > 1e-30 else 0.0

    print("\nPerformance (flow-weighted outlet):")
    print(f"η_Cl2 (top/total)  = {eta_cl2*100:.2f}%")
    print(f"η_CO  (bottom/total)= {eta_co*100:.2f}%")
    print(f"Cl2 in bottom frac  = {cl2_in_bottom_frac*100:.3f}%")
    print(f"CO  in top frac     = {co_in_top_frac*100:.3f}%")
    print(f"CO2 conversion      = {co2_conv:.1f}%")

    print("\nSteady-state diagnostics (control-volume mass balance):")
    print(f"Cl2: steps={steps_cl2}, relMB={diag_cl2.get('mb_rel', float('nan')):.2e}")
    print(f"CO:  steps={steps_co},  relMB={diag_co.get('mb_rel', float('nan')):.2e}")
    print(f"CO2: steps={steps_co2}, relMB={diag_co2.get('mb_rel', float('nan')):.2e}")

    # For CO2, also report whether the requested consumption flux was limited by availability
    if "flux_bot_req" in diag_co2 and "flux_bot_eff_mean" in diag_co2:
        req = float(diag_co2["flux_bot_req"])
        eff = float(diag_co2["flux_bot_eff_mean"])
        limited_frac = float(diag_co2.get("flux_bot_limited_frac", float("nan")))
        if abs(req) > 0:
            print(f"CO2 wall flux: requested={req:.3e}, effective_mean={eff:.3e} (|eff|/|req|={abs(eff/req):.2f}), limited_x_frac={limited_frac:.2f}")

    return {
        "name": name,
        "geom": geom,
        "flow": flow,
        "elec": elec,
        "dx": dx,
        "dy": dy,
        "x": x,
        "y": y,
        "vx": vx,
        "v_avg": v_avg,
        "v_max": v_max,
        "Re": Re,
        "dt": dt,
        "Npp": Npp,
        "fields": {"Cl2": C_Cl2, "CO": C_CO, "CO2": C_CO2},
        "outlet": {"Cl2": m_cl2, "CO": m_co, "CO2": m_co2},
        "eta": {"Cl2": eta_cl2, "CO": eta_co},
        "xcontam": {"Cl2_in_bottom": cl2_in_bottom_frac, "CO_in_top": co_in_top_frac},
        "co2_conversion_pct": co2_conv,
        "diagnostics": {"Cl2": diag_cl2, "CO": diag_co, "CO2": diag_co2},
        "steps": {"Cl2": steps_cl2, "CO": steps_co, "CO2": steps_co2},
    }


def plot_comparison(base: dict[str, object], opt: dict[str, object], outpath: str = "comparison_base_vs_optimal_fixed.png") -> None:
    geom_b: Geometry = base["geom"]  # type: ignore[assignment]
    geom_o: Geometry = opt["geom"]  # type: ignore[assignment]

    x_b = base["x"]
    y_b = base["y"]
    x_o = opt["x"]
    y_o = opt["y"]

    Cb = base["fields"]
    Co = opt["fields"]

    XX_b, YY_b = np.meshgrid(x_b, y_b, indexing="ij")
    XX_o, YY_o = np.meshgrid(x_o, y_o, indexing="ij")

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Base Case vs Optimal Design (FIXED)", fontsize=14, fontweight="bold")

    # Base row
    cf0 = axs[0, 0].contourf(XX_b * 1000, YY_b * 1000, Cb["Cl2"], 30, cmap="viridis")
    axs[0, 0].set_title(f"Base: Cl₂\nη={base['eta']['Cl2']*100:.2f}%")
    axs[0, 0].set_ylabel("y (mm)")
    axs[0, 0].axhline(y=0, color="white", linestyle="--", linewidth=1)

    cf1 = axs[0, 1].contourf(XX_b * 1000, YY_b * 1000, Cb["CO"], 30, cmap="viridis")
    axs[0, 1].set_title(f"Base: CO\nη={base['eta']['CO']*100:.2f}%")
    axs[0, 1].axhline(y=0, color="white", linestyle="--", linewidth=1)

    cf2 = axs[0, 2].contourf(XX_b * 1000, YY_b * 1000, Cb["CO2"], 30, cmap="viridis")
    axs[0, 2].set_title(f"Base: CO₂\nConv={base['co2_conversion_pct']:.1f}%")
    axs[0, 2].axhline(y=0, color="white", linestyle="--", linewidth=1)

    # Optimal row
    cf3 = axs[1, 0].contourf(XX_o * 1000, YY_o * 1000, Co["Cl2"], 30, cmap="viridis")
    axs[1, 0].set_title(f"Opt: Cl₂\nη={opt['eta']['Cl2']*100:.2f}%")
    axs[1, 0].set_xlabel("x (mm)")
    axs[1, 0].set_ylabel("y (mm)")
    axs[1, 0].axhline(y=0, color="white", linestyle="--", linewidth=1)

    cf4 = axs[1, 1].contourf(XX_o * 1000, YY_o * 1000, Co["CO"], 30, cmap="viridis")
    axs[1, 1].set_title(f"Opt: CO\nη={opt['eta']['CO']*100:.2f}%")
    axs[1, 1].set_xlabel("x (mm)")
    axs[1, 1].axhline(y=0, color="white", linestyle="--", linewidth=1)

    cf5 = axs[1, 2].contourf(XX_o * 1000, YY_o * 1000, Co["CO2"], 30, cmap="viridis")
    axs[1, 2].set_title(f"Opt: CO₂\nConv={opt['co2_conversion_pct']:.1f}%")
    axs[1, 2].set_xlabel("x (mm)")
    axs[1, 2].axhline(y=0, color="white", linestyle="--", linewidth=1)

    for cf, ax in zip([cf0, cf1, cf2, cf3, cf4, cf5], axs.ravel()):
        plt.colorbar(cf, ax=ax)

    plt.tight_layout()
    plt.savefig(outpath, dpi=250, bbox_inches="tight")

    print("\n" + "=" * 72)
    print("PLOTS")
    print("=" * 72)
    print(f"Saved: {outpath}")


def main() -> None:
    # Inlets
    C0_Cl2 = 1e-6
    C0_CO = 0.0
    C0_CO2 = 0.03 * 1000.0  # 30 mol/m^3

    diff = Diffusivities()

    # Base-case parameters (as in the paste)
    base_geom = Geometry(L=0.8, H=0.40e-3, Nx=220, Ny=80)
    base_flow = Flow(v_areal=1.5e-4)
    base_elec = Electrochem(j=600.0)

    base = run_case(
        name="BASE",
        geom=base_geom,
        flow=base_flow,
        elec=base_elec,
        diff=diff,
        C0_Cl2=C0_Cl2,
        C0_CO=C0_CO,
        C0_CO2=C0_CO2,
        verbose_solver=True,
    )

    # "Optimal" parameters (as in the paste)
    opt_geom = Geometry(L=0.95, H=0.25e-3, Nx=220, Ny=80)
    opt_flow = Flow(v_areal=0.8e-4)
    opt_elec = Electrochem(j=1000.0)

    opt = run_case(
        name="OPTIMAL",
        geom=opt_geom,
        flow=opt_flow,
        elec=opt_elec,
        diff=diff,
        C0_Cl2=C0_Cl2,
        C0_CO=C0_CO,
        C0_CO2=C0_CO2,
        verbose_solver=True,
    )

    # Comparison table
    print("\n" + "=" * 72)
    print("BASE vs OPTIMAL (FIXED / CONSISTENT)")
    print("=" * 72)
    print(f"{'Metric':<28} {'Base':>12} {'Opt':>12} {'Req':>12}")
    print("-" * 72)
    print(f"{'η_Cl2 (%)':<28} {base['eta']['Cl2']*100:>12.2f} {opt['eta']['Cl2']*100:>12.2f} {'>95':>12}")
    print(f"{'η_CO (%)':<28}  {base['eta']['CO']*100:>12.2f} {opt['eta']['CO']*100:>12.2f} {'>95':>12}")
    print(f"{'CO2 conversion (%)':<28} {base['co2_conversion_pct']:>12.1f} {opt['co2_conversion_pct']:>12.1f} {'~95':>12}")
    print(f"{'Re':<28} {base['Re']:>12.1f} {opt['Re']:>12.1f} {'<1000':>12}")

    # Constraint check
    def ok(name: str, cond: bool) -> str:
        return f"{'✓' if cond else '✗'} {name}"

    print("\nConstraints (optimal):")
    print(ok("L < 1 m", opt_geom.L < 1.0))
    print(ok("H < 1 mm", opt_geom.H < 1e-3))
    print(ok("Re < 1000", float(opt['Re']) < 1000.0))
    print(ok("η_Cl2 > 95%", float(opt['eta']['Cl2']) > 0.95))
    print(ok("η_CO > 95%", float(opt['eta']['CO']) > 0.95))

    plot_comparison(base, opt)


if __name__ == "__main__":
    main()
