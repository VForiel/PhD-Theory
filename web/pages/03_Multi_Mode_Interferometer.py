"""
Streamlit page for Multi-Mode Interferometer (MMI) simulation and analysis.

This page provides an interactive interface to:
- Explore MMI geometry and optical properties
- Run simulations with custom input amplitudes and phases
- Calibrate input phases to optimize null depth
- Visualize field propagation and output characteristics

Adapted from HELIOS examples/14_mmi_streamlit.py for PHISE.
"""

import sys
from pathlib import Path
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import importlib

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent
# Ensure project root is on path so `src` is importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import phise.modules.mmi as mmi_module

# Ensure latest version of the plotting helpers (Streamlit keeps modules cached)
importlib.reload(mmi_module)

from phise.modules.mmi import (
    simulate,
    calibrate_input_phases_genetic,
    calibrate_n_core_and_phases,
    plot_mmi_interactive,
)

def render_calibration_result(result: dict):
    """Render calibration metrics and phase evolution plot."""
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Best Null/Bright Ratio", f"{result['best_metric']:.2e}")
        st.write("**Optimized phases [rad]:**")
        for i, phi in enumerate(result['best_phases']):
            st.write(f"  Input {i+1}: {phi:.4f} rad ({phi*180/np.pi:.1f}°)")

    with col2:
        metric_array = np.asarray(result.get('metric', []), dtype=float)
        phases_array = np.asarray(result.get('phases', []), dtype=float)

        if len(metric_array) > 0:
            fig, axes = plt.subplots(2, 1, figsize=(9, 8))

            ax = axes[0]
            ax.semilogy(metric_array, 'o-', lw=2, markersize=4, color='steelblue')
            ax.axhline(y=result['best_metric'], color='red', linestyle='--', lw=2, alpha=0.7, label='Best')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Null/Bright Ratio')
            ax.set_title('Phase Calibration Convergence')
            ax.grid(True, alpha=0.3)
            ax.legend()

            ax2 = axes[1]
            if phases_array.size > 0:
                for k in range(phases_array.shape[1]):
                    ax2.plot(np.arange(len(phases_array)), phases_array[:, k]/np.pi, lw=1.8, label=f'Input {k+1}')
                ax2.set_ylabel('Phase (×π rad)')
                ax2.set_xlabel('Iteration')
                ax2.set_title('Phase Evolution')
                ax2.grid(True, alpha=0.3)
                ax2.legend(ncol=2, fontsize=8)
            else:
                ax2.text(0.5, 0.5, 'No phase history available', ha='center', va='center')

            plt.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)

st.set_page_config(
    page_title="MMI Simulation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Multi-Mode Interferometer 🔀")

st.markdown(r"""
## Overview

A **Multi-Mode Interferometer (MMI)** is a photonic device that combines light from 
multiple input ports through constructive and destructive interference to produce outputs 
with specific amplitude and phase relationships.

### Physical Principles

When light from N inputs propagates through the MMI region of width **W** and length **L**,
it decomposes into eigenmodes that evolve with different propagation constants:

$$\beta_m = \sqrt{(k_0 n_\text{core})^2 - (m\pi/W)^2}$$

where:
- **k₀** = 2π/λ is the free-space wavenumber
- **n_core** is the core refractive index
- **m** is the mode number (1, 2, 3, ...)

The output ports couple to the field at z=L, producing complex amplitudes that depend on:
- **Input amplitudes** and **phases**
- **Waveguide geometry** (N, M, W, L)
- **Material properties** (n_core, Δn)
- **Operating wavelength** (λ)

### Key Applications

- **Phase routing**: Direct light to specific outputs
- **Power splitting**: Distribute input power to multiple ports
- **Nulling interferometry**: Suppress companion starlight for exoplanet imaging
- **Photonic circuits**: Building blocks for integrated optics
""")

st.divider()

# =======================
# MMI CONFIGURATION
# =======================

st.markdown("## MMI Configuration")

col_geom1, col_geom2 = st.columns(2)

with col_geom1:
    st.subheader("Geometry")
    N = st.number_input("Inputs (N)", 1, 8, 4)
    M = st.number_input("Outputs (M)", 1, 8, 4)
    W_um = st.number_input("Width W (µm)", value=20.0, min_value=1.0, max_value=100.0, step=0.5)
    W = W_um * 1e-6
    L_um = st.number_input("Length L (µm) — 0 = auto", value=440.0, min_value=0.0, step=10.0)
    L = (L_um * 1e-6) if L_um > 0 else None

with col_geom2:
    st.subheader("Optics")
    wavelength_um = st.number_input("Wavelength λ (µm)", value=1.55, min_value=0.4, max_value=4.0, step=0.01, format="%.2f")
    wavelength = wavelength_um * 1e-6
    n_core = st.number_input("Core Index (n_core)", min_value=1.0, max_value=4.0, value=float(st.session_state.get("n_core_override", 2.0458)), format="%.4f")
    delta_n = st.number_input("Index Contrast (Δn)", 0.001, 1.0, 0.1, format="%.4f")

col_num1, col_num2 = st.columns([1,2])
with col_num1:
    st.subheader("Numerical")


    num_modes = st.number_input("Number of modes", value=200, min_value=10, max_value=200, step=1)

    z_mode = st.radio("Z-resolution mode", ["Number of pixels", "Resolution (µm)"])
    
    if z_mode == "Resolution (µm)":
        z_resolution_um = st.number_input("Z-resolution (µm) — 0 = auto", value=0.0, min_value=0.0, step=0.01, format="%.3f")
        z_resolution = (z_resolution_um * 1e-6) if z_resolution_um > 0 else None
    else:
        # Calculate effective L for display
        L_display = L if L is not None else 0.0
        num_z_pixels = st.number_input("Number of Z pixels", value=500, min_value=10, max_value=10000, step=100)
        # Calculate z_resolution from L and num_pixels
        if L is not None and L > 0:
            z_resolution = L / num_z_pixels
        else:
            z_resolution = None  # Will be auto-calculated with default L

with col_num2:
    st.subheader("Waveguides")

    col_guides1, col_guides2 = st.columns(2)

    with col_guides1:
        st.write("**Inputs**")
        Din_um = st.number_input("Spacing Din (µm) — 0 = auto", value=5.0, min_value=0.0, max_value=50.0, step=0.5)
        Din = (Din_um * 1e-6) if Din_um > 0 else None
        
        Sin_um = st.number_input("Width Sin (µm) — 0 = auto", value=4.5, min_value=0.0, max_value=10.0, step=0.1)
        Sin = (Sin_um * 1e-6) if Sin_um > 0 else None

    with col_guides2:
        st.write("**Outputs**")
        Dout_um = st.number_input("Spacing Dout (µm) — 0 = auto", value=5.0, min_value=0.0, max_value=50.0, step=0.5)
        Dout = (Dout_um * 1e-6) if Dout_um > 0 else None
        
        Sout_um = st.number_input("Width Sout (µm) — 0 = auto", value=4.5, min_value=0.0, max_value=10.0, step=0.1)
        Sout = (Sout_um * 1e-6) if Sout_um > 0 else None

    bright_idx = st.number_input("Bright output index", min_value=0, max_value=max(0, int(M)-1), value=0, step=1)

st.divider()

# =======================
# INPUT CONFIGURATION
# =======================

st.markdown("## Input Configuration")

st.subheader("Amplitudes & Phases")

# Apply staged phase updates from calibration before rendering sliders (wrap to [0, 2π])
if "phase_updates" in st.session_state:
    updates = st.session_state.pop("phase_updates")
    for i, val in enumerate(updates[: int(N)]):
        wrapped = float(np.mod(val, 2.0))  # val is in units of π, keep in [0,2]
        st.session_state[f"phase_{i}"] = wrapped

# Initialize session state for sliders
for i in range(int(N)):
    if f"amp_{i}" not in st.session_state:
        st.session_state[f"amp_{i}"] = 1.0 if i == 0 else 1.0
    if f"phase_{i}" not in st.session_state:
        st.session_state[f"phase_{i}"] = 0.0

# Collect input values
amps = []
phases = []
phase_cols = st.columns(int(N))

for i in range(int(N)):
    with phase_cols[i]:
        st.write(f"**Input {i+1}**")
        amp = st.slider(
            f"Amplitude", 0.0, 2.0, st.session_state[f"amp_{i}"], 0.05,
            key=f"amp_{i}", label_visibility="collapsed"
        )
        phase_pi = st.slider(
            f"Phase (×π rad)", 0.0, 2.0, st.session_state[f"phase_{i}"], 0.01,
            key=f"phase_{i}", label_visibility="collapsed"
        )
        amps.append(amp)
        phases.append(phase_pi * np.pi)

# Convert to complex amplitudes
input_amplitudes = np.array([amp * np.exp(1j * phi) for amp, phi in zip(amps, phases)])

# Normalize if needed
total_power = np.sum(np.abs(input_amplitudes)**2)
if total_power > 0:
    input_amplitudes_normalized = input_amplitudes / np.sqrt(total_power)
else:
    input_amplitudes_normalized = input_amplitudes

st.divider()

# =======================
# SIMULATION CONTROLS
# =======================

st.markdown("## Simulation Controls")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ Run Simulation", width="stretch"):
        st.session_state.pop("calibration_result", None)
        st.session_state.pop("show_calibration", None)
        st.session_state.pop("ncore_result", None)
        st.session_state.pop("show_ncore", None)
        st.session_state.run_sim = True

with col2:
    if st.button("🎯 Calibrate Phases", width="stretch"):
        st.session_state.pop("calibration_result", None)
        st.session_state.pop("show_calibration", None)
        st.session_state.pop("ncore_result", None)
        st.session_state.pop("show_ncore", None)
        st.session_state.calibrate_phases = True

with col3:
    if st.button("⚙️ Optimize n_core", width="stretch"):
        st.session_state.pop("calibration_result", None)
        st.session_state.pop("show_calibration", None)
        st.session_state.pop("ncore_result", None)
        st.session_state.pop("show_ncore", None)
        st.session_state.optimize = True

# =======================
# SIMULATION EXECUTION
# =======================

if st.session_state.get("run_sim"):
    st.info("Running MMI simulation...")
    
    try:
        # Run simulation
        output_amps = simulate(
            N=int(N), M=int(M), L=L, W=W, n_core=n_core, delta_n=delta_n,
            wavelength=wavelength, input_amplitudes=input_amplitudes,
            Din=Din, Dout=Dout, Sin=Sin, Sout=Sout,
            num_modes=num_modes, z_resolution=z_resolution, verbose=False
        )
        
        # Display results
        st.success("✓ Simulation complete")
        
        # Create visualization
        fig = plot_mmi_interactive(
            N=int(N), M=int(M), L=L, W=W, n_core=n_core, delta_n=delta_n,
            wavelength=wavelength, input_amplitudes=input_amplitudes,
            Din=Din, Dout=Dout, Sin=Sin, Sout=Sout,
            num_modes=num_modes, num_z_steps=None, z_resolution=z_resolution,
            verbose=False
        )
        
        st.pyplot(fig, width="stretch")
        
        # Output analysis
        st.subheader("Output Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Output Amplitudes**")
            for j in range(int(M)):
                amp = output_amps[j]
                st.write(f"Output {j+1}: {np.abs(amp):.4f} ∠ {np.angle(amp)*180/np.pi:.1f}°")
        
        with col2:
            st.write("**Output Powers**")
            powers = np.abs(output_amps)**2
            for j in range(int(M)):
                st.write(f"Output {j+1}: {powers[j]:.4f}")
        
        with col3:
            st.write("**Power Distribution**")
            total_out = np.sum(powers)
            for j in range(int(M)):
                frac = powers[j] / total_out * 100 if total_out > 0 else 0
                st.write(f"Output {j+1}: {frac:.1f}%")
        
        # Null depth (if 2 outputs)
        if M >= 2:
            st.divider()
            null_out = np.sort(powers)[0]  # Minimum power
            bright_out = np.sort(powers)[-1]  # Maximum power
            
            if bright_out > 0:
                null_depth = null_out / bright_out
                null_depth_mag = -2.5 * np.log10(null_depth) if null_depth > 0 else np.inf
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Null/Bright Ratio", f"{null_depth:.2e}")
                with col2:
                    st.metric("Null Depth", f"{null_depth_mag:.2f} mag" if np.isfinite(null_depth_mag) else "∞")
        
        plt.close(fig)
        st.session_state.run_sim = False
        
    except Exception as e:
        st.error(f"❌ Simulation failed: {str(e)}")
        st.session_state.run_sim = False


elif st.session_state.get("calibrate_phases"):
    st.info("Calibrating input phases to minimize null depth...")
    
    try:
        # Use normalized magnitudes
        magnitudes = np.abs(input_amplitudes_normalized)
        
        # Run calibration
        result = calibrate_input_phases_genetic(
            N=int(N), M=int(M), L=L, W=W, n_core=n_core, delta_n=delta_n,
            wavelength=wavelength, input_amplitudes=magnitudes,
            Din=Din, Dout=Dout, Sin=Sin, Sout=Sout,
            bright_output_idx=int(bright_idx), num_modes=num_modes,
            z_resolution=z_resolution, verbose=False
        )
        
        st.success("✓ Calibration complete")

        st.session_state["calibration_result"] = {
            "best_metric": float(result["best_metric"]),
            "best_phases": [float(phi) for phi in result["best_phases"]],
            "metric": [float(x) for x in result.get("metric", [])],
            "phases": np.asarray(result.get("phases", []), dtype=float).tolist(),
        }
        st.session_state["show_calibration"] = True
        st.session_state.calibrate_phases = False
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Calibration failed: {str(e)}")
        st.session_state.calibrate_phases = False


elif st.session_state.get("optimize"):
    st.info("Optimizing n_core and input phases…")

    # Progress UI
    prog = st.progress(0.0, text="Stage 1: coarse scan")
    status1 = st.empty()
    status2 = st.empty()

    def cb_coarse(cur, total):
        frac = min(max(float(cur) / float(total), 0.0), 1.0)
        prog.progress(frac, text=f"Stage 1: coarse scan {cur}/{total}")
        status1.write(f"Stage 1: {cur}/{total}")

    def cb_grad(iter_idx, delta):
        status2.write(f"Stage 2: iteration {iter_idx}, Δn_core = {delta:.4f}")

    try:
        result = calibrate_n_core_and_phases(
            N=int(N), M=int(M), L=L, W=W,
            n_core_initial=float(n_core),
            n_core_min=1.0,
            n_core_max=2.0*float(n_core),
            delta_n=delta_n,
            wavelength=wavelength,
            input_amplitudes=np.abs(input_amplitudes_normalized),
            bright_output_idx=int(bright_idx),
            num_modes=num_modes,
            num_z_steps=30,
            z_resolution=None,
            Din=Din, Dout=Dout, Sin=Sin, Sout=Sout,
            n_core_steps_coarse=20,
            gradient_convergence_threshold=1e-3,
            gradient_initial_step=0.01,
            beta=0.8,
            initial_step=np.pi/2,
            epsilon=1e-3,
            verbose=False,
            progress_callback_coarse=cb_coarse,
            progress_callback_gradient=cb_grad,
        )

        prog.progress(1.0, text="Stage 1 complete")
        st.success("✓ n_core optimization complete")

        # Persist results and rerun for unified rendering below
        st.session_state["ncore_result"] = {
            "n_core_values_coarse": np.asarray(result.get("n_core_values_coarse", [])).tolist(),
            "metrics_coarse": np.asarray(result.get("metrics_coarse", [])).tolist(),
            "n_core_values_gradient": np.asarray(result.get("n_core_values_gradient", [])).tolist(),
            "metrics_gradient": np.asarray(result.get("metrics_gradient", [])).tolist(),
            "best_n_core": float(result["best_n_core"]),
            "best_metric": float(result["best_metric"]),
            "best_phases": np.asarray(result["best_phases"], dtype=float).tolist(),
        }
        st.session_state["show_ncore"] = True
        st.session_state.optimize = False
        st.rerun()

    except Exception as e:
        st.error(f"❌ n_core optimization failed: {str(e)}")
        st.session_state.optimize = False

# =======================
# CALIBRATION DISPLAY (persistent until another action)
# =======================

if st.session_state.get("show_calibration") and st.session_state.get("calibration_result"):
    st.subheader("Calibration Results")
    result = st.session_state["calibration_result"]
    render_calibration_result(result)

    if st.button("📋 Apply calibrated phases to inputs", key="apply_phases_persistent"):
        phase_updates = [float(np.mod(phi, 2 * np.pi) / np.pi) for phi in result["best_phases"]]
        st.session_state["phase_updates"] = phase_updates
        st.session_state["show_calibration"] = True
        st.rerun()

# =======================
# N_CORE OPTIMIZATION DISPLAY (persistent)
# =======================

if st.session_state.get("show_ncore") and st.session_state.get("ncore_result"):
    st.subheader("n_core Optimization Results")
    res = st.session_state["ncore_result"]

    # Two-panel figure: coarse scan and gradient refinement
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: coarse scan
    ncc = np.asarray(res.get("n_core_values_coarse", []), dtype=float)
    mc = np.asarray(res.get("metrics_coarse", []), dtype=float)
    if ncc.size > 0 and mc.size > 0:
        ax = axes[0]
        ax.semilogy(ncc, mc, "o-", lw=2, markersize=6, color="coral", label="Coarse Scan")
        ax.axvline(x=res["best_n_core"], color="red", linestyle="--", lw=2, label=f"Best: {res['best_n_core']:.4f}")
        ax.set_xlabel("n_core")
        ax.set_ylabel("Null/Bright Ratio")
        ax.set_title("Stage 1: Coarse Grid Scan")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    # Right: gradient refinement
    ng = np.asarray(res.get("n_core_values_gradient", []), dtype=float)
    mg = np.asarray(res.get("metrics_gradient", []), dtype=float)
    if ng.size > 0 and mg.size > 0:
        ax = axes[1]
        ax.semilogy(range(len(mg)), mg, "s-", lw=2, markersize=6, color="mediumseagreen", label="Gradient")
        ax.axhline(y=res["best_metric"], color="red", linestyle="--", lw=2, label=f"Best: {res['best_metric']:.3e}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Null/Bright Ratio")
        ax.set_title("Stage 2: Gradient Descent")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        if len(mg) > 0:
            best_idx = int(np.argmin(mg))
            ax.annotate(
                f"Min @ iter {best_idx}",
                xy=(best_idx, mg[best_idx]),
                xytext=(best_idx + max(1, len(mg)//10), mg[best_idx] * 1.8),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
                fontsize=9,
                color="red",
            )

    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # Best summary and apply button
    colb1, colb2 = st.columns(2)
    with colb1:
        st.metric("Best n_core", f"{res['best_n_core']:.4f}")
        st.metric("Best Null/Bright", f"{res['best_metric']:.3e}")
    with colb2:
        st.write("**Optimized phases [rad]:**")
        for i, phi in enumerate(res["best_phases"]):
            st.write(f"  Input {i+1}: {phi:.4f} rad ({phi*180/np.pi:.1f}°)")

    if st.button("📋 Apply best n_core and phases", key="apply_ncore_phases"):
        st.session_state["n_core_override"] = float(res["best_n_core"])
        st.session_state["phase_updates"] = [float(np.mod(phi, 2*np.pi) / np.pi) for phi in res["best_phases"]]
        st.session_state["show_ncore"] = True
        st.rerun()

# =======================
# DOCUMENTATION
# =======================

st.divider()

with st.expander("ℹ️ About This Tool", expanded=False):
    st.markdown("""
    This MMI simulator is part of **PHISE** (PHotonic Interferometry Simulation for Exoplanets),
    a Python package for end-to-end simulation of interferometric instruments.
    
    **Imported from:** HELIOS MMI module (github.com/VForiel/HELIOS)
    
    **Author:** Vincent Foriel (2025)
    
    **Features:**
    - Eigenmode expansion with step-index modes
    - Mode-dependent effective indices
    - Phase calibration for nulling
    - Interactive visualization
    
    **For more information:** Visit the [HELIOS documentation](https://helios-project.readthedocs.io/en/latest/api/sim/mmi.html)
    """)