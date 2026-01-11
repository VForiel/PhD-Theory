"""
Reusable context configurator widget for Streamlit pages.

Provides a dropdown/expander-based context editor with preset templates
and full parameter configuration.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional, Callable
from copy import deepcopy as copy

import numpy as np
import pandas as pd
import streamlit as st
import astropy.units as u

ROOT = Path(__file__).parent.parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phise import Context
from phise.classes import Interferometer, Telescope, Camera, Target, Companion
from phise.classes.archs import SuperKN


def context_widget(
    key_prefix: str = "ctx",
    presets: Optional[Dict[str, Context]] = None,
    default_preset: str = "VLTI",
    expanded: bool = False,
    show_advanced: bool = True,
    initial_context: Optional[Context] = None,
) -> Context:
    """
    Render a full context configurator widget with all parameters.

    Args:
        key_prefix: Unique prefix for widget keys (to avoid collisions).
        presets: Dictionary of preset contexts {name: Context}. If None, uses VLTI/LIFE.
        default_preset: Name of the default preset to load initially.
        expanded: Whether the expander should be open by default.
        show_advanced: Show advanced chip/camera/telescope settings.
        initial_context: Context object to use as default initialization.

    Returns:
        Context: Fully configured context object.
    """

    # Initialize presets
    if presets is None:
        presets = {
            "VLTI": Context.get_VLTI(),
            "LIFE": Context.get_LIFE(),
        }

    def load_preset_logic(preset_name: str) -> Context:
        """Helper to load and modify a preset."""
        return copy(presets.get(preset_name, Context.get_VLTI()))

    # Session state key
    ctx_key = f"{key_prefix}_context"
    if ctx_key not in st.session_state:
        if initial_context is not None:
            st.session_state[ctx_key] = copy(initial_context)
        else:
            st.session_state[ctx_key] = load_preset_logic(default_preset)

    with st.expander("⚙️ Context Configuration", expanded=expanded):
        # Preset selector
        preset_cols = st.columns([2, 1, 1])
        with preset_cols[0]:
            selected_preset = st.selectbox(
                "Preset",
                options=list(presets.keys()),
                index=list(presets.keys()).index(default_preset) if default_preset in presets else 0,
                key=f"{key_prefix}_preset",
            )
        with preset_cols[1]:
            if st.button("Load Preset", key=f"{key_prefix}_load", width="stretch"):
                st.session_state[ctx_key] = load_preset_logic(selected_preset)
                st.rerun()
        with preset_cols[2]:
            if st.button("Reset to Default", key=f"{key_prefix}_reset", width="stretch"):
                st.session_state[ctx_key] = load_preset_logic(default_preset)
                st.rerun()

        ctx = copy(st.session_state[ctx_key])

        # Tabs for different configuration sections
        tabs = st.tabs(["📍 Observation", "🔭 Interferometer", "✨ Kernel-Nuller", "📷 Camera", "🎯 Target"])
        
        # ===== TAB 1: Observation Parameters =====
        with tabs[0]:
            st.caption("Observation Settings")
            obs_cols = st.columns(2)
            
            with obs_cols[0]:
                # ctx_name removed

                h_val = st.slider(
                    "Hour angle h (hours)",
                    min_value=-12.0,
                    max_value=12.0,
                    value=float(ctx.h.to(u.hourangle).value),
                    step=0.1,
                    key=f"{key_prefix}_h",
                )
                gamma_val = st.number_input(
                    "Cophasing RMS Γ (nm)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=float(ctx.Γ.to(u.nm).value),
                    step=1.0,
                    key=f"{key_prefix}_gamma",
                )
            
            with obs_cols[1]:
                dh_val = st.slider(
                    "Observation span Δh (hours)",
                    min_value=0.1,
                    max_value=24.0,
                    value=float(ctx.Δh.to(u.hourangle).value),
                    step=0.1,
                    key=f"{key_prefix}_dh",
                )
                mono = st.checkbox(
                    "Monochromatic approximation",
                    value=ctx.monochromatic,
                    key=f"{key_prefix}_mono",
                )

        # ===== TAB 2: Interferometer =====
        with tabs[1]:
            st.caption("Interferometer Configuration")
            
            # Basic parameters
            int_cols1 = st.columns(5)
            with int_cols1[0]:
                lat_val = st.number_input(
                    "Latitude l (deg)",
                    min_value=-90.0,
                    max_value=90.0,
                    value=float(ctx.interferometer.l.to(u.deg).value),
                    step=0.1,
                    key=f"{key_prefix}_lat",
                )
            with int_cols1[1]:
                lambda_val = st.number_input(
                    "Wavelength λ (µm)",
                    min_value=0.2,
                    max_value=20.0,
                    value=float(ctx.interferometer.λ.to(u.um).value),
                    step=0.01,
                    key=f"{key_prefix}_lambda",
                )
            with int_cols1[2]:
                bandwidth_val = st.number_input(
                    "Bandwidth Δλ (µm)",
                    min_value=0.001,
                    max_value=10.0,
                    value=float(ctx.interferometer.Δλ.to(u.um).value),
                    step=0.01,
                    key=f"{key_prefix}_bandwidth",
                )
            with int_cols1[3]:
                fov_val = st.number_input(
                    "Field of view (mas)",
                    min_value=0.1,
                    max_value=1000.0,
                    value=float(ctx.interferometer.fov.to(u.mas).value),
                    step=1.0,
                    key=f"{key_prefix}_fov",
                )
            with int_cols1[4]:
                eta_val = st.number_input(
                    "Efficiency η",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(ctx.interferometer.η),
                    step=0.01,
                    key=f"{key_prefix}_eta",
                )
            
            # Telescopes configuration
            if show_advanced:
                st.divider()
                st.caption("Telescope Array")
                tel_data = [
                    {
                        "Name": t.name,
                        "x (m)": float(t.r.to(u.m).value[0]),
                        "y (m)": float(t.r.to(u.m).value[1]),
                        "Area (m²)": float(t.a.to(u.m**2).value),
                    }
                    for t in ctx.interferometer.telescopes
                ]
                edited_tel = st.data_editor(
                    pd.DataFrame(tel_data),
                    width="stretch",
                    key=f"{key_prefix}_tel_editor",
                )
            
        # ===== TAB 3: Kernel-Nuller =====
        with tabs[2]:
            st.caption("Photonic Component (SuperKN)")
            
            kn_cols = st.columns(3)
            with kn_cols[0]:
                lambda0_val = st.number_input(
                    "Design wavelength λ₀ (µm)",
                    min_value=0.2,
                    max_value=20.0,
                    value=float(ctx.interferometer.chip.λ0.to(u.um).value),
                    step=0.01,
                    key=f"{key_prefix}_lambda0",
                )
            with kn_cols[1]:
                phi_mean = st.number_input(
                    "Injected phase mean (nm)",
                    min_value=-1000.0,
                    max_value=1000.0,
                    value=float(np.mean(ctx.interferometer.chip.φ.to(u.nm).value)),
                    step=1.0,
                    key=f"{key_prefix}_phi_mean",
                )
            with kn_cols[2]:
                sigma_mean = st.number_input(
                    "Manufacturing error std (nm)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(np.std(ctx.interferometer.chip.σ.to(u.nm).value)),
                    step=1.0,
                    key=f"{key_prefix}_sigma_std",
                )

        # ===== TAB 4: Camera =====
        with tabs[3]:
            st.caption("Camera Settings")
            
            cam_cols = st.columns(2)
            with cam_cols[0]:
                # camera_name removed

                exp_minutes = st.number_input(
                    "Exposure time (min)",
                    min_value=0.01,
                    value=float(ctx.interferometer.camera.e.to(u.min).value),
                    step=0.1,
                    key=f"{key_prefix}_exp",
                )

            with cam_cols[1]:
                camera_ideal = st.checkbox(
                    "Ideal (noise-free) camera",
                    value=ctx.interferometer.camera.ideal,
                    key=f"{key_prefix}_camera_ideal",
                )

        # ===== TAB 5: Target =====
        with tabs[4]:
            st.caption("Target Configuration")
            
            # Target basic parameters
            target_cols = st.columns(3)
            with target_cols[0]:
                target_name = st.text_input(
                    "Target name",
                    value=ctx.target.name,
                    key=f"{key_prefix}_target_name",
                )
            with target_cols[1]:
                flux_mode = st.selectbox(
                    "Flux Input Unit",
                    options=["Magnitude (AB)", "Flux Density (Jy)", "Irradiance (W/m²/nm)"],
                    index=1, # Default to Jy as it is more common than raw SI for this field
                    key=f"{key_prefix}_flux_mode",
                )
                
                # Get current flux in various units for display/initialization
                current_f = ctx.target.f
                # Conversion logic
                # W/m²/nm <-> Jy requires wavelength.
                # 1 Jy = 10^-26 W/m^2/Hz
                # f_lambda = f_nu * c / lambda^2
                lam = ctx.interferometer.λ
                
                if flux_mode == "Magnitude (AB)":
                    # Convert current f (W/m^2/nm) -> Jy -> ABmag
                    f_jy = current_f.to(u.Jy, equivalencies=u.spectral_density(lam))
                    current_mag = f_jy.to(u.ABmag).value
                    
                    mag_val = st.number_input(
                        "Magnitude (AB)",
                        value=float(current_mag),
                        step=0.1,
                        key=f"{key_prefix}_target_mag",
                    )
                    # Update ctx.target.f
                    # Mag -> Jy -> W/m^2/nm
                    new_flux_jy = (mag_val * u.ABmag).to(u.Jy)
                    ctx.target.f = new_flux_jy.to(u.W / u.m**2 / u.nm, equivalencies=u.spectral_density(lam))
                    
                elif flux_mode == "Flux Density (Jy)":
                    f_jy = current_f.to(u.Jy, equivalencies=u.spectral_density(lam)).value
                    jy_val = st.number_input(
                        "Flux Density (Jy)",
                        min_value=0.0,
                        value=float(f_jy),
                        step=10.0,
                        format="%.2f",
                        key=f"{key_prefix}_target_jy",
                    )
                    # Jy -> W/m^2/nm
                    ctx.target.f = (jy_val * u.Jy).to(u.W / u.m**2 / u.nm, equivalencies=u.spectral_density(lam))
                    
                else: # Irradiance
                    exp_val = st.number_input(
                        "Spectral Flux (W/m²/nm)",
                        min_value=0.0,
                        value=float(current_f.value),
                        step=1e-15,
                        format="%.2e", # Scientific notation
                        key=f"{key_prefix}_target_flux",
                    )
                    ctx.target.f = exp_val * (u.W / u.m**2 / u.nm)
            with target_cols[2]:
                target_dec = st.number_input(
                    "Declination δ (deg)",
                    min_value=-90.0,
                    max_value=90.0,
                    value=float(ctx.target.δ.to(u.deg).value),
                    step=0.1,
                    key=f"{key_prefix}_target_dec",
                )
            
            # Companions
            st.divider()
            st.caption("Companions")
            comp_data = [
                {
                    "Name": c.name,
                    "Contrast": float(c.c),
                    "Separation (mas)": float(c.ρ.to(u.mas).value),
                    "Polar Angle (0°=East)": float(c.θ.to(u.deg).value),
                }
                for c in ctx.target.companions
            ]
            # Prepare DataFrame with explicit types
            if comp_data:
                df_comp = pd.DataFrame(comp_data)
            else:
                df_comp = pd.DataFrame(columns=["Name", "Contrast", "Separation (mas)", "Polar Angle (0°=East)"])
            
            # Ensure numeric columns are float
            for col in ["Contrast", "Separation (mas)", "Polar Angle (0°=East)"]:
                if col in df_comp.columns:
                     df_comp[col] = df_comp[col].astype(float)

            comp_edited = st.data_editor(
                df_comp,
                column_config={
                    "Contrast": st.column_config.NumberColumn(
                        "Contrast",
                        format="%.2e",
                        min_value=0.0,
                        max_value=1.0,
                        # step=None # Allow free input
                    )
                },
                num_rows="dynamic",
                width="stretch",
                key=f"{key_prefix}_comp_editor",
            )

        # Build updated context
        try:
            # Build telescopes
            if show_advanced and 'edited_tel' in locals():
                telescopes = []
                for _, row in edited_tel.iterrows():
                    name = str(row.get("Name", "Telescope"))
                    x = float(row.get("x (m)", 0.0))
                    y = float(row.get("y (m)", 0.0))
                    area = float(row.get("Area (m²)", 1.0))
                    telescopes.append(Telescope(a=area * u.m**2, r=np.array([x, y]) * u.m, name=name))
            else:
                telescopes = ctx.interferometer.telescopes

            # Build chip
            if show_advanced:
                phi_vals = np.ones(14) * phi_mean * u.nm
                sigma_vals = np.abs(np.random.normal(0, sigma_mean, 14)) * u.nm
                chip = SuperKN(φ=phi_vals, σ=sigma_vals, λ0=lambda0_val * u.um, name=ctx.interferometer.chip.name)
            else:
                chip = ctx.interferometer.chip

            # Build camera
            camera = Camera(e=exp_minutes * u.min, ideal=camera_ideal, name=ctx.interferometer.camera.name)

            # Build interferometer
            interferometer = Interferometer(
                l=lat_val * u.deg,
                λ=lambda_val * u.um,
                Δλ=bandwidth_val * u.um,
                fov=fov_val * u.mas,
                η=eta_val,
                telescopes=telescopes,
                chip=chip,
                camera=camera,
                name=ctx.interferometer.name,
            )

            # Build companions
            companions = []
            for _, row in comp_edited.iterrows():
                if pd.notna(row.get("Name")):
                    cname = str(row.get("Name", "Companion"))
                    contrast = float(row.get("Contrast", 0.0))
                    sep_mas = float(row.get("Separation (mas)", 0.0))
                    pa_deg = float(row.get("Polar Angle (0°=East)", 0.0))
                    companions.append(Companion(c=contrast, ρ=sep_mas * u.mas, θ=pa_deg * u.deg, name=cname))

            # Build target
            target = Target(
                f=ctx.target.f, # Use the updated flux from the UI block above
                δ=target_dec * u.deg,
                companions=companions,
                name=target_name,
            )

            # Build context
            new_ctx = Context(
                interferometer=interferometer,
                target=target,
                h=h_val * u.hourangle,
                Δh=dh_val * u.hourangle,
                Γ=gamma_val * u.nm,
                monochromatic=mono,
                name=ctx.name,
            )

            # Save to session state
            st.session_state[ctx_key] = new_ctx

        except Exception as e:
            st.error(f"Error building context: {e}")
            st.session_state[ctx_key] = ctx

    return st.session_state[ctx_key]


def simple_context_selector(
    key_prefix: str = "ctx",
    presets: Optional[Dict[str, Context]] = None,
    default_preset: str = "VLTI",
) -> Context:
    """
    Minimal context selector - just a dropdown and quick parameters.

    Args:
        key_prefix: Unique prefix for widget keys.
        presets: Dictionary of preset contexts.
        default_preset: Default preset name.

    Returns:
        Context: Selected/modified context.
    """

    if presets is None:
        presets = {
            "VLTI": Context.get_VLTI(),
            "LIFE": Context.get_LIFE(),
        }

    ctx_key = f"{key_prefix}_context"
    if ctx_key not in st.session_state:
        st.session_state[ctx_key] = copy(presets.get(default_preset, Context.get_VLTI()))

    # Compact layout
    st.caption("Context Configuration")
    top_cols = st.columns([2, 2, 1, 1])

    with top_cols[0]:
        selected_preset = st.selectbox(
            "Preset",
            options=list(presets.keys()),
            index=list(presets.keys()).index(default_preset) if default_preset in presets else 0,
            key=f"{key_prefix}_preset",
            label_visibility="collapsed",
        )

    with top_cols[1]:
        h_val = st.slider(
            "Hour angle (h)",
            min_value=-12.0,
            max_value=12.0,
            value=float(st.session_state[ctx_key].h.to(u.hourangle).value),
            step=0.1,
            key=f"{key_prefix}_h",
            label_visibility="collapsed",
        )

    with top_cols[2]:
        gamma_val = st.number_input(
            "Γ (nm)",
            min_value=0.0,
            value=float(st.session_state[ctx_key].Γ.to(u.nm).value),
            step=10.0,
            key=f"{key_prefix}_gamma",
            label_visibility="collapsed",
        )

    with top_cols[3]:
        if st.button("Load", key=f"{key_prefix}_load", width="stretch"):
            st.session_state[ctx_key] = copy(presets[selected_preset])
            st.rerun()

    # Update context
    ctx = copy(st.session_state[ctx_key])
    ctx.h = h_val * u.hourangle
    ctx.Γ = gamma_val * u.nm
    st.session_state[ctx_key] = ctx

    return ctx
