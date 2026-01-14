import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
import astropy.units as u
from phise import Context, Companion
from copy import deepcopy
import sys
from pathlib import Path

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent
# Add src to path for phise if needed
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Add web to path for utils
sys.path.insert(0, str(ROOT / "web"))
from utils.context_widget import context_widget

st.set_page_config(page_title="Modulation Spectrum", page_icon="📉", layout="wide")

st.title("Modulation Spectrum 📉")

st.markdown("""
### Frequency Analysis

As we have seen, the time-domain signal is dominated by noise and mixing.
The elegant solution proposed in this thesis is to move to the **frequency domain**.

By calculating the Power Spectral Density (PSD) of the time series, we exploit the fact that:
1.  **The planet has a periodic signature**: Rotation induces modulation at a specific fundamental frequency.
2.  **Noise Separation**: Instrumental noise (pistons) lives mainly at high frequencies or as 1/f (low frequency), leaving a clear window for the planetary signal.
3.  **Contrast Independence**: The spectral peak appears at the same frequency regardless of contrast (only its height changes).
4.  **Spatial Disentanglement (Source Separation)**: Two planets at different positions will have different modulation frequencies, appearing as distinct peaks.

#### Disadvantages
*   **Spectral Spreading**: If the modulation is not a pure sinusoid (harmonics), energy is spread over multiple peaks, reducing the signal-to-noise ratio of the main peak. This happens especially far from the star (fast and complex modulation). However, close to the Star, the modulation is slow and quasi-sinusoidal, which is ideal for this method. This is perfect because it is precisely the zone most difficult to access for other techniques!
""")

st.success("""
##### ✨ My Contribution
Application of the Modulation Spectrum to Kernel-Nulling for multi-planet detection. I demonstrated that this frequency analysis can be used to disentangle signals from multiple planets in the same field.
""")

# --- Context Configuration ---
st.subheader("Configuration")

presets = {
    "VLTI": Context.get_VLTI(),
    "LIFE": Context.get_LIFE(),
}

base_ctx = context_widget(
    key_prefix="mod_spec",
    presets=presets,
    default_preset="VLTI",
    expanded=False, 
    show_advanced=True
)

# Force ideal component parameters (φ=0, σ=0) for this simulation
base_ctx.interferometer.chip.φ = np.zeros(14) * u.nm
base_ctx.interferometer.chip.σ = np.zeros(14) * u.nm
st.info("ℹ️ **Simulation Note:** The photonic component is forced to be **ideal** (no phase errors, no manufacturing defects) for this demonstration.")

# --- Simulation Controls (Main Layout) ---

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    st.markdown("**Companion Parameters (Override)**")
    nb_planets = st.slider("Number of simulated planets", 1, 3, 1)
    
    companions = []
    # Create mini-columns for each planet to save space
    for i in range(nb_planets):
        with st.expander(f"Planet {i+1}", expanded=(i==0)):
            rho = st.slider(f"P{i+1} Separation (mas)", 0.1, 15.0, 2.0 * (i+1), 0.1)
            theta = st.slider(f"P{i+1} Polar Angle (0°=East)", 0.0, 360.0, float((45 * (i+1)) % 360), 1.0)
            contrast = st.slider(f"P{i+1} Contrast (Log10)", -8.0, -2.0, -4.0, 0.1)
            companions.append({"rho": rho, "theta": theta, "c": 10**contrast})

with col_sim2:
    st.markdown("**Noise & Observation**")
    gamma_nm = st.slider("Atmospheric Piston RMS (nm)", 0.0, 100.0, 10.0, 1.0, help="Piston noise (Gamma)")
    delta_h = st.slider("Observation Duration (Δh)", 1, 24, 6, 1, help="Period over which kernels are optimized/valid.")
    k_idx = st.selectbox("Kernel Index", options=[0, 1, 2], index=0, format_func=lambda x: f"Kernel {x+1}", help="Index of the kernel output to analyze.")
    
    st.caption("The simulation generates a time series based on the selected context, then computes its periodogram.")


# --- Processing ---



# Valid setup - Base Context
ctx_user = deepcopy(base_ctx)

# Selection of signals to display
available_signals = ["Total"] + [f"Planet {i+1}" for i in range(nb_planets)]
selected_signals = st.multiselect("Select Signals to Display", available_signals, default=["Total"])

# Helper to configure context for a specific signal scenario
def configuration_context_for_signal(base, signal_name, all_companions):
    ctx_copy = deepcopy(base)
    ctx_copy.target.companions = []
    
    if signal_name == "Total":
        # Add all planets
        for comp in all_companions:
            ctx_copy.target.companions.append(
                Companion(c=comp["c"], ρ=comp["rho"]*u.mas, θ=comp["theta"]*u.deg)
            )
    else:
        # Extract planet index (e.g. "Planet 1" -> index 0)
        idx = int(signal_name.split(" ")[1]) - 1
        comp = all_companions[idx]
        ctx_copy.target.companions = [
            Companion(c=comp["c"], ρ=comp["rho"]*u.mas, θ=comp["theta"]*u.deg)
        ]
    return ctx_copy

def run_simulation_logic(ctx_input, _gamma, _delta_h, _duration, is_ideal=False):
    # Simulation Parameters
    fs_inv = 2 * u.minute 
    fs = (1/fs_inv).to(u.Hz).value
    
    n_points = int((_duration * 60) / 2)
    
    # Calculate time range centered on the context's current h
    h_center = ctx_input.h.to(u.hourangle).value
    h_start = h_center - _duration/2
    h_end = h_center + _duration/2
    h_array = np.linspace(h_start, h_end, n_points)
    
    fluxes = []
    
    # Use a local copy for scanning time
    sim_ctx = deepcopy(ctx_input)
    # Apply Delta H Override
    sim_ctx.Δh = _delta_h * u.hourangle 
    
    # Apply conditions
    if is_ideal:
        # Force ideal parameters
        sim_ctx.monochromatic = True
        sim_ctx.interferometer.chip.σ = np.zeros(14) * u.nm
        sim_ctx.Γ = 0 * u.nm
        sim_ctx.interferometer.camera.ideal = True
    else:
        # Use user parameters + Gamma + Realistic Camera
        sim_ctx.Γ = _gamma * u.nm
        sim_ctx.interferometer.camera.ideal = False # Enables Photon Noise in observe()

    for h in h_array:
        sim_ctx.h = h * u.hourangle
        raw = sim_ctx.observe() 
        k = sim_ctx.interferometer.chip.process_outputs(raw)
        fluxes.append(k)
        
    fluxes = np.array(fluxes) # (n_points, nb_kernels)
    
    # Noise is already included in observe() if camera.ideal is False
    return h_array, fluxes, fs


# --- Run Simulations & Collect Results ---
results = {} # Key: signal_name, Value: (h, sig_ideal, sig_obs, psd_ideal, psd_obs, freqs)

fs_hz_global = None # Just to reuse common freq axis if needed

for signal_name in selected_signals:
    # 1. Setup Context
    ctx_scenario = configuration_context_for_signal(ctx_user, signal_name, companions)
    
    # 2. Run Ideal
    # We use delta_h as both the Context range and the Simulation duration
    h_arr, f_ideal, fs_hz = run_simulation_logic(ctx_scenario, 0, delta_h, delta_h, is_ideal=True)
    
    # 3. Run Observed
    h_arr, f_obs, fs_hz = run_simulation_logic(ctx_scenario, gamma_nm, delta_h, delta_h, is_ideal=False)
    
    fs_hz_global = fs_hz
    
    # 4. Process Spectrum
    # Safety check for kernel index
    n_kernels = f_ideal.shape[1]
    if k_idx >= n_kernels:
        st.warning(f"Selected Kernel Index {k_idx} is out of bounds (Max: {n_kernels-1}). Showing Index 0.")
        valid_k_idx = 0
    else:
        valid_k_idx = k_idx

    s_ideal = f_ideal[:, valid_k_idx]
    s_obs = f_obs[:, valid_k_idx]
    
    results[signal_name] = {
        "h": h_arr,
        "s_ideal": s_ideal,
        "s_obs": s_obs,
    }


# --- Display ---

st.divider()

col_plots = st.container()

with col_plots:
    # 1. Time Series
    st.subheader("Time Domain")
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    
    colors = ['C0', 'C1', 'C2', 'C3', 'C4'] # Colors for different signals
    
    for i, (name, res) in enumerate(results.items()):
        c = colors[i % len(colors)]
        # Plot Ideal
        ax1.plot(res['h'], res['s_ideal'], color=c, linestyle='--', label=f"{name} (Ideal)", alpha=0.6, linewidth=1.5)
        # Plot Observed
        ax1.plot(res['h'], res['s_obs'], color=c, linestyle='-', label=f"{name} (Observed)", alpha=0.9, linewidth=1.0)
    
    ax1.set_xlabel("Hour Angle (h)")
    ax1.set_ylabel("Signal")
    if len(results) > 0:
        first_res = list(results.values())[0]
        ax1.set_xlim(first_res['h'][0], first_res['h'][-1])
    ax1.legend(ncol=len(results) if len(results) < 4 else 4, fontsize='small')
    st.pyplot(fig1)
    
    # 2. Spectrum
    col_header, col_toggle = st.columns([0.65, 0.35])
    col_header.subheader("Frequency Domain (Spectrum)")
    
    with col_toggle:
        col_log, col_win = st.columns(2)
        use_log = col_log.checkbox("Log Scale", value=False)
        use_window = col_win.checkbox("Hann Window", value=False, help="Apply Hann window to reduce spectral leakage (side lobes) but slightly widen the main peak.")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    
    window_type = 'hann' if use_window else 'boxcar'
    
    max_peak_freq = 0.0
    
    for i, (name, res) in enumerate(results.items()):
        
        # Re-compute spectrum with chosen window
        s_ideal = res['s_ideal']
        s_obs = res['s_obs']
        fs_hz = fs_hz_global # we can reuse fs as it is constant
        nfft = len(s_obs) * 8
        
        freqs, psd_ideal = periodogram(s_ideal, fs=fs_hz, window=window_type, nfft=nfft)
        freqs, psd_obs = periodogram(s_obs, fs=fs_hz, window=window_type, nfft=nfft)
        
        c = colors[i % len(colors)]
        freqs_cph = freqs * 3600
        
        # Slice [1:] to remove DC component (idx 0)
        ax2.plot(freqs_cph[1:], psd_ideal[1:], color=c, linestyle='--', label=f"{name} (Ideal)", linewidth=1.5)
        ax2.plot(freqs_cph[1:], psd_obs[1:], color=c, linestyle='-', label=f"{name} (Obs)", alpha=0.8, linewidth=1.0)
        
        # If it's a specific planet, mark its main peaks
        if name.startswith("Planet"):
            # Find peaks in the ideal spectrum (clean reference)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(psd_ideal[1:], height=np.max(psd_ideal[1:])/100) # Prominence treshold
            
            # Sort by height to keep top 3
            peak_heights = psd_ideal[1:][peaks]
            sorted_indices = np.argsort(peak_heights)[::-1][:3]
            top_peaks = peaks[sorted_indices]
            
            for p_idx in top_peaks:
                freq_val = freqs_cph[1:][p_idx]
                max_peak_freq = max(max_peak_freq, freq_val)
                ax2.axvline(freq_val, color=c, linestyle=':', alpha=0.5, linewidth=1)
    
    ax2.set_xlabel("Frequency (cycles / hour)")
    ax2.set_ylabel("Power Spectral Density")
    
    # Dynamic Limit: 1.5x the highest detected peak
    if max_peak_freq > 0:
        ax2.set_xlim(0, max_peak_freq * 1.5)
    else:
        ax2.set_xlim(0, 1) # Focus on low freq (planets are slow)
    if use_log:
        ax2.set_yscale('log')
    else:
        ax2.set_yscale('linear')
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(ncol=len(results) if len(results) < 4 else 4, fontsize='small')
    
    st.pyplot(fig2)

# Conclusions
st.success("""
**Interpretation**:
*   Even if the time signal seems purely noisy (especially with low contrast), clear peaks emerge in the spectrum.
*   Each planet generates its own fundamental peak (and sometimes harmonics).
*   White noise remains a flat background "carpet" in the spectrum, not masking the peaks if integration is sufficient.
""")

with st.expander("ℹ️ Technical Details: Signal Processing"):
    st.markdown("""
    **1. Periodogram (PSD)**:
    We use the periodogram to estimate the Power Spectral Density of the signal. It essentially computes the squared magnitude of the Discrete Fourier Transform (DFT).
    
    **2. Zero Padding**:
    To improve the visual smoothness of the spectrum, we use "Zero Padding". We append zeros to the end of the time-series signal before computing the FFT (here, `nfft = 8 * len(signal)`). This interpolates the spectrum, adding more points between the fundamental frequency bins, but **it does not increase the true resolution** (which depends only on the observation duration `Δh`).
    
    **3. Hann Window (Apodization)**:
    Since our observation window is rectangular (we observe for duration `Δh` then stop), the spectrum is convolved with a Sinc function, creating "side lobes" or "bumps" around the main peaks (Spectral Leakage).
    *   **Hann Window**: We multiply the time signal by a bell-shaped curve (Hann window) that goes to zero at the edges. This smooths out the sharp transitions, effectively suppressing the side lobes.
    *   **Trade-off**: The side lobes disappear, but the main peak becomes slightly wider (loss of resolution).
    """)
