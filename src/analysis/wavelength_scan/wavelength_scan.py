"""Analyse et balayage en longueur d'onde.

Fonctions pour simuler et afficher la réponse du nuller en fonction de
la longueur d'onde.
"""
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
try:
    plt.rcParams['image.origin'] = 'lower'
except Exception:
    pass
from copy import deepcopy as copy
import astropy.units as u
from scipy import stats
from phise.classes.context import Context
from phise.modules import utils
from io import BytesIO

def run(ctx: Context=None, scan_range=0.2 * u.um, obs_bandwidth=0 * u.um, n=11, figsize=(5, 5), save_as=None, return_image=False, algo="Obstruction", algo_params=None, progress_callback=None):

    if algo_params is None:
        algo_params = {}

    # 1. Ensure n is odd (centered on λ0)
    if n % 2 == 0:
        n += 1

    # 2. Build Base Context
    if ctx is None:
        ctx = Context.get_VLTI()
        ctx.interferometer.chip.σ = np.zeros(14) * u.m  # Default ideal if None
    
    base_ctx = copy(ctx)
    base_ctx.Γ = 0 * u.nm
    base_ctx.target.companions = [] # No companions, observe star only for null depth
    
    # We reset the phases to ensure we start "fresh", though calibration should overwrite them.
    # We DO NOT reset sigma (σ) because that's the error we are trying to characterize/correct.
    # If the user really meant sigma=0, the result would be perfect and uninteresting.
    base_ctx.interferometer.chip.φ *= 0 

    λ0 = base_ctx.interferometer.chip.λ0.to(u.um)
    λs = np.linspace(λ0.value - scan_range.value / 2, λ0.value + scan_range.value / 2, n) * u.um
    
    data_dynamic = np.zeros(n)
    data_static = np.zeros(n) # Will be filled when we hit λ0

    plt.figure(figsize=figsize)
    plt.axvline(λ0.to(u.nm).value, color='k', ls='--', label='$\\lambda_0$')

    # 3. Main Loop (Wavelength Scan)
    for i, λ_current in enumerate(λs):
        
        progress = i / n
        msg = f'⌛ Calibrating at {round(λ_current.value, 3)} um...'
        
        if progress_callback:
            progress_callback(progress, msg)
        else:
            print(f"{msg} {round(progress * 100, 2)}%", end='\r')
            
        # --- A. Prepare Contexts ---
        
        # ctx_cal: Monochromatic, for finding optimal phases
        ctx_cal = copy(base_ctx)
        ctx_cal.monochromatic = True
        ctx_cal.interferometer.Δλ = 1e-6 * u.nm
        ctx_cal.interferometer.λ = λ_current
        
        # ctx_obs: Potentially Polychromatic, for measuring performance
        ctx_obs = copy(base_ctx)
        ctx_obs.interferometer.λ = λ_current
        if obs_bandwidth.to(u.nm).value > 1e-3:
            ctx_obs.monochromatic = False
            ctx_obs.interferometer.Δλ = obs_bandwidth
        else:
            ctx_obs.monochromatic = True
            ctx_obs.interferometer.Δλ = 1e-6 * u.nm
            
        # --- B. Calibrate (on ctx_cal) ---
        if algo == "Genetic":
            ctx_cal.calibrate_gen(β=algo_params.get('beta', 0.961), verbose=False)
        else: # Obstruction
            ctx_cal.calibrate_obs(n=algo_params.get('n_samples', 1000))
            
        # --- C. Transfer Phases to Observation Context ---
        # The calibration updated ctx_cal.interferometer.chip.φ
        ctx_obs.interferometer.chip = copy(ctx_cal.interferometer.chip)
        
        # --- D. Observe (Dynamic Point) ---
        outs = ctx_obs.observe()
        k = ctx_obs.interferometer.chip.process_outputs(outs)
        b = outs[0]
        data_dynamic[i] = np.mean(np.abs(k) / b)
        
        # --- E. Static Curve Generation (at λ0) ---
        # If we are at (or very close to) λ0, we use THIS calibration to simulate the static curve
        if np.isclose(λ_current.value, λ0.value, atol=1e-5):
            
            # We use the current ctx_obs state (which has phases optimized for λ0)
            # and scan it across all wavelengths without recalibrating.
            
            ctx_static = copy(ctx_obs) # Has λ0 calibration
            
            for j, λ_stat in enumerate(λs):
                ctx_static.interferometer.λ = λ_stat
                outs_s = ctx_static.observe() # Observe with frozen phases
                k_s = ctx_static.interferometer.chip.process_outputs(outs_s)
                b_s = outs_s[0]
                data_static[j] = np.mean(np.abs(k_s) / b_s)
            
            plt.plot(λs.to(u.nm).value, data_static, color='gray', alpha=0.3, label='$\\lambda_{cal} = \\lambda_0$')

    # 4. Finalize
    if progress_callback:
        progress_callback(1.0, "✅ Done.")
    else:
        print(f"✅ Done.{' ' * 30}")
        
    plt.plot(λs.to(u.nm).value, data_dynamic, 'o-', label='$\\lambda_{cal} = \\lambda$')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Mean Kernel-Null Depth')
    plt.yscale('log')
    plt.title('Spectral Scan Analysis')
    plt.legend()
    
    if save_as:
        utils.save_plot(save_as, "wavelength_scan.png")
    
    if return_image:
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        return buf.getvalue()
        
    plt.show()