"""Visualisations et transformations des distributions issues des
observations du kernel-nuller.

Fonctions pour calculer et tracer la distribution instantanée, les
évolutions temporelles et autres représentations des données.
"""
import numpy as np
import astropy.units as u
from copy import deepcopy
import matplotlib.pyplot as plt
try:
    plt.rcParams['image.origin'] = 'lower'
except Exception:
    pass
from phise import Context
from phise.classes.archs.superkn import expected_outputs_jit
from phise.modules import utils

π = np.pi

def compute_analytical_distrib(n, ctx, opd_errors, α, β, φ1, φ2, φ3, φ4):
    """
    Compute analytical distribution using SuperKN.expected_outputs_jit.
    """
    if ctx.monochromatic:
        λ_range = np.array([ctx.interferometer.λ.to(u.m).value])
    else:
        λ0 = ctx.interferometer.λ.to(u.m).value
        Δλ = ctx.interferometer.Δλ.to(u.m).value
        λ_range = np.linspace(λ0 - Δλ/2, λ0 + Δλ/2, 5)
    
    λ0 = ctx.interferometer.λ.to(u.m).value
    
    brights = np.zeros(n)
    darks = np.zeros((n, 6))
    kernels = np.zeros((n, 3))
    
    geometric_phases = np.array([φ1, φ2, φ3, φ4])
    
    for i in range(n):
        b_acc = 0
        d_acc = np.zeros(6)
        k_acc = np.zeros(3)
        
        for λ in λ_range:
            # Compute phases
            ph = geometric_phases * (λ0 / λ)
            sig = 2 * np.pi * opd_errors[i] / λ
            
            # Construct ψ (input fields)
            # Star field per input j: sqrt(α)/4 * exp(1j * σ_j)
            # Planet field per input j: sqrt(β)/4 * exp(1j * (σ_j + φ_j))
            
            # Incoherent addition: calculate outputs for Star and Planet separately
            
            # Star
            ψ_s = np.zeros(4, dtype=complex)
            for j in range(4):
                ψ_s[j] = np.sqrt(α/4) * np.exp(1j * sig[j])
            b_s, d_s, k_s = expected_outputs_jit(ψ_s)

            # Planet
            ψ_p = np.zeros(4, dtype=complex)
            for j in range(4):
                ψ_p[j] = np.sqrt(β/4) * np.exp(1j * (sig[j] + ph[j]))
            b_p, d_p, k_p = expected_outputs_jit(ψ_p)
            
            # Sum intensities
            b_acc += b_s + b_p
            d_acc += d_s + d_p
            k_acc += k_s + k_p
            
        brights[i] = b_acc / len(λ_range)
        darks[i] = d_acc / len(λ_range)
        kernels[i] = k_acc / len(λ_range)
        
    return brights, darks, kernels

#==============================================================================
# Instantaneous distribution
#==============================================================================

def instant_distribution(
        ctx: Context=None,
        n=10000,
        stat=np.median,
        figsize=(10, 10),
        log=False,
        density = False, 
        sync_plots = True,
        save_path=None,
        show=True,
        save_as=None,
        return_details: bool = False,
        verbose: bool = False
    ) -> np.ndarray:
    """
    Get the instantaneous distribution of the kernel nuller.

    Parameters
    ----------
    ctx : Context
        The context to use.
    n : int, optional
        The number of samples to take, by default 1000
    stat : function, optional
        The function to use to compute the statistic, by default np.median.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    show : bool, optional
        Whether to show the plot. Default is True.
    save_as : str, optional
        Path or directory to save the figures. If provided, figures are saved with automatic naming if it's a directory.
    return_details : bool, optional
        If True, return a dictionary with all numerical arrays and contexts for reuse
        (e.g., for analytical verification). Defaults to False for backward compatibility.
    verbose : bool, optional
        If True, print summary statistics and signal info. Default is True (legacy behavior).

    Returns
    -------
    np.ndarray
        The instantaneous distribution of the kernel nuller.
    """

    # Set up contexts ---------------------------------------------------------

    if ctx is None:
        ctx = Context.get_VLTI()
        ctx.interferometer.chip.σ = np.zeros(14) * u.um
        ctx.target.companions[0].c = 0.1
    else:
        ctx = deepcopy(ctx)
        if ctx.target.companions == []:
            raise ValueError('No companion in the context. Please add a companion to the target.')
    ctx.Δh = ctx.interferometer.camera.e.to(u.hour).value * u.hourangle

    ctx_so = deepcopy(ctx)
    ctx_so.target.companions = []

    ctx_po = deepcopy(ctx)
    scale = 1e12
    ctx_po.target.f /= scale # Scale down the star flux to make it negligible
    ctx_po.target.companions[0].c *= scale

    if verbose:
        print(f"Star: Signal: {np.sum(ctx_po.pf) * ctx_po.interferometer.camera.e:.2e}, flux : {ctx_po.target.f:.2e}")
        print(f"Planet: Contrast: {ctx_po.target.companions[0].c:.2e} Signal: {np.sum(ctx_po.pf) * ctx_po.target.companions[0].c * ctx_po.interferometer.camera.e:.2e}")

    # Numerical model ---------------------------------------------------------

    # Prepare data arrays
    data = np.empty((n, 3))
    data_so = np.empty((n, 3))
    data_po = np.empty((n, 3))

    darks = np.empty((n, 6))
    darks_so = np.empty((n, 6))
    darks_po = np.empty((n, 6))

    brights = np.empty(n)
    brights_so = np.empty(n)
    brights_po = np.empty(n)

    errors = np.empty((n, 4)) * ctx.Γ.unit

    # Sample data
    for i in range(n):

        # Generate noise
        upstream_pistons = np.random.normal(0, ctx.Γ.value, size=len(ctx.interferometer.telescopes)) * ctx.Γ.unit
        errors[i, :] = upstream_pistons

        # Distrib with companion(s)
        outs = ctx.observe(upstream_pistons=upstream_pistons)
        data[i, :] = ctx.interferometer.chip.process_outputs(outs)
        brights[i] = outs[0]
        darks[i, :] = outs[1:]
        
        # Distrib with star only
        outs_so = ctx_so.observe(upstream_pistons=upstream_pistons)
        data_so[i, :] = ctx_so.interferometer.chip.process_outputs(outs_so)
        brights_so[i] = outs_so[0]
        darks_so[i, :] = outs_so[1:]

        # Distrib with planet only
        outs_po = ctx_po.observe(upstream_pistons=upstream_pistons)
        data_po[i, :] = ctx_po.interferometer.chip.process_outputs(outs_po)
        brights_po[i] = outs_po[0]
        darks_po[i, :] = outs_po[1:]


    # Plotting (numerical only) ------------------------------------------------

    scenarios = [data_so, data_po, data]
    scenario_titles = ['Star only', 'Planet only', 'Star + planet']

    for i, numerical_data in enumerate(scenarios):

        (_, axs) = plt.subplots(3, 3, figsize=figsize, constrained_layout=True, sharex=sync_plots)
        plt.suptitle(scenario_titles[i], fontsize=16)
        
        # Get plot limits
        lim = 0
        for k in range(3):

            keep = 50
            percentil = (100 - keep) / 2

            # Get x limits
            (xmin_plot, xmax_plot) = np.percentile(numerical_data[:, k], [percentil, keep + percentil])
            lim = max(lim, abs(xmin_plot), abs(xmax_plot))

        # Get bins
        if sync_plots:
            bins = np.linspace(-lim, lim, 2 * int(np.sqrt(n)) + 1)
        else:
            bins = 2 * int(np.sqrt(n)) + 1

        # Plot histograms
        for k in range(3):
            for j in range(3):
                axs[k, j].hist(numerical_data[:, k], label=r'$\mathcal{S}(s+p)$', bins=bins, alpha=0.5, color='blue', density=density, log=log)
                axs[k, j].axvline(stat(numerical_data[:,k]), color='blue', linestyle='--', alpha=0.5)
                axs[k,j].set_ylabel('Occurrences (%)')
                axs[k,j].set_title(f'Kernel {k + 1}')
            if sync_plots:
                axs[k,0].set_xlim(-lim, lim)
                axs[k,2].set_xlim(-lim, lim)
        axs[2,0].set_xlabel('Kernel output')
        axs[2,1].set_xlabel('Kernel output')
        axs[2,2].set_xlabel('Kernel output')
        
        # Auto-save logic
        if save_as:
            scenario_names = ['star_only', 'planet_only', 'full']
            utils.save_plot(save_as, f"instant_distribution_{scenario_names[i]}.png")
        elif save_path:
            fname = save_path
            if i > 0:
                import os
                base, ext = os.path.splitext(save_path)
                fname = f"{base}_{i}{ext}"
            plt.savefig(fname)
            
        if show:
            plt.show()
        else:
            plt.close()

    # Basic stats for numerical data only ------------------------------------
    if verbose:
        print("\n# Statistiques numériques (instantanées)")
    scenarios_stats = {
        "Étoile seule": {
            "bright": brights_so,
            "darks": darks_so,
            "kernels": data_so
        },
        "Planète seule": {
            "bright": brights_po,
            "darks": darks_po,
            "kernels": data_po
        },
        "Complet": {
            "bright": brights,
            "darks": darks,
            "kernels": data
        }
    }
    if verbose:
        print("\n## Total Photons (Bright + Sum(Darks))")
        for name, data_dict in scenarios_stats.items():
            num_b = data_dict["bright"]
            num_d = data_dict["darks"]
            total_num = np.mean(num_b) + np.sum(np.mean(num_d, axis=0))
            print(f"- {name}: {total_num:.3e}")

        print("\n## Bright")
        for stat_name, stat_func in [("Moyenne", np.mean), ("Médiane", np.median), ("Std", np.std)]:
            print(f"\n### {stat_name}")
            for name, data_dict in scenarios_stats.items():
                num = data_dict["bright"]
                print(f"- {name}: {stat_func(num):.3e}")

        for k in range(6):
            print(f"\n## Dark {k+1}")
            for stat_name, stat_func in [("Moyenne", np.mean), ("Médiane", np.median), ("Std", np.std)]:
                print(f"\n### {stat_name}")
                for name, data_dict in scenarios_stats.items():
                    num = data_dict["darks"]
                    print(f"- {name}: {stat_func(num[:, k]):.3e}")

        for k in range(3):
            print(f"\n## Kernel {k+1}")
            for stat_name, stat_func in [("Moyenne", np.mean), ("Médiane", np.median), ("Std", np.std)]:
                print(f"\n### {stat_name}")
                for name, data_dict in scenarios_stats.items():
                    num = data_dict["kernels"]
                    print(f"- {name}: {stat_func(num[:, k]):.3e}")

    if return_details:
        return {
            "data": data,
            "data_so": data_so,
            "data_po": data_po,
            "brights": brights,
            "brights_so": brights_so,
            "brights_po": brights_po,
            "darks": darks,
            "darks_so": darks_so,
            "darks_po": darks_po,
            "errors": errors,
            "ctx": ctx,
            "ctx_so": ctx_so,
            "ctx_po": ctx_po
        }

    return (data, data_so)


def verify_distributions(
        ctx: Context=None,
        n=10000,
        stat=np.median,
        figsize=(10, 10),
        log=False,
        density=False,
        sync_plots=True,
        save_path=None,
        show=True,
        save_as=None,
        verbose: bool = False
    ):
    """
    Compute numerical distributions (via `instant_distribution`) and compare them
    to analytical distributions for verification.

    Parameters mirror `instant_distribution`; numerical plots are suppressed in
    this routine to avoid double rendering.
    """

    # Run numerical sampling once to reuse the same OPD errors for analytics
    details = instant_distribution(
        ctx=ctx,
        n=n,
        stat=stat,
        figsize=figsize,
        log=log,
        density=density,
        sync_plots=sync_plots,
        save_path=None,
        show=False,
        save_as=None,
        return_details=True,
        verbose=verbose
    )

    ctx = details["ctx"]
    ctx_so = details["ctx_so"]
    ctx_po = details["ctx_po"]

    data = details["data"]
    data_so = details["data_so"]
    data_po = details["data_po"]
    errors = details["errors"]

    # Analytical model --------------------------------------------------------
    ψi = ctx.get_input_fields()
    φ1, φ2, φ3, φ4 = np.angle(ψi[1])
    e = ctx.interferometer.camera.e.to(u.s).value
    α = np.sum(ctx.pf).value * e
    β = α * ctx.target.companions[0].c
    opd_errors = errors.to(u.m).value

    α_so = np.sum(ctx_so.pf).value * e
    β_so = 0

    α_po = np.sum(ctx_po.pf).value * e
    β_po = α_po * ctx_po.target.companions[0].c

    b_comb, d_comb, k_comb = compute_analytical_distrib(n, ctx, opd_errors, α, β, φ1, φ2, φ3, φ4)
    analytical_brights = b_comb
    analytical_darks = d_comb
    analytical_data = k_comb
    
    b_so, d_so, k_so = compute_analytical_distrib(n, ctx_so, opd_errors, α_so, β_so, φ1, φ2, φ3, φ4)
    analytical_brights_so = b_so
    analytical_darks_so = d_so
    analytical_data_so = k_so
    
    b_po, d_po, k_po = compute_analytical_distrib(n, ctx_po, opd_errors, α_po, β_po, φ1, φ2, φ3, φ4)
    analytical_brights_po = b_po
    analytical_darks_po = d_po
    analytical_data_po = k_po

    # Plotting comparison ----------------------------------------------------

    scenario_full = [data, analytical_data]
    scenario_star_only = [data_so, analytical_data_so]
    scenario_planet_only = [data_po, analytical_data_po]

    for i, (numerical_data, analytical_data) in enumerate([scenario_star_only, scenario_planet_only, scenario_full]):

        (_, axs) = plt.subplots(3, 3, figsize=figsize, constrained_layout=True, sharex=sync_plots)
        plt.suptitle(['Star only', 'Planet only', 'Star + planet'][i], fontsize=16)
        
        # Get plot limits
        lim = 0
        for k in range(3):

            keep = 50
            percentil = (100 - keep) / 2

            # Get x limits
            (xmin_plot, xmax_plot) = np.percentile(numerical_data[:, k], [percentil, keep + percentil])
            lim = max(lim, abs(xmin_plot), abs(xmax_plot))
            (xmin_plot, xmax_plot) = np.percentile(analytical_data[:, k], [percentil, keep + percentil])
            lim = max(lim, abs(xmin_plot), abs(xmax_plot))

        # Get bins
        if sync_plots:
            bins = np.linspace(-lim, lim, 2 * int(np.sqrt(n)) + 1)
        else:
            bins = 2 * int(np.sqrt(n)) + 1

        # Plot histograms
        for k in range(3):
            for j in range(3):
                if j in [1,2]:
                    axs[k, j].hist(analytical_data[:,k], label=r'$\mathcal{A}(s+p)$', bins=bins, alpha=0.5, color='green', density=density, log=log)
                    axs[k, j].axvline(stat(analytical_data[:,k]), color='green', linestyle='--', alpha=0.5)
                if j in [0,2]:
                    axs[k, j].hist(numerical_data[:, k], label=r'$\mathcal{S}(s+p)$', bins=bins, alpha=0.5, color='blue', density=density, log=log)
                    axs[k, j].axvline(stat(numerical_data[:,k]), color='blue', linestyle='--', alpha=0.5)
                axs[k,j].set_ylabel('Occurrences (%)')
                axs[k,j].set_title(f'Kernel {k + 1}')
            if sync_plots:
                axs[k,0].set_xlim(-lim, lim)
                axs[k,2].set_xlim(-lim, lim)
        axs[2,0].set_xlabel('Kernel output')
        axs[2,1].set_xlabel('Kernel output')
        axs[2,2].set_xlabel('Kernel output')
        
        # Auto-save logic
        if save_as:
            scenario_names = ['star_only', 'planet_only', 'full']
            utils.save_plot(save_as, f"verify_distributions_{scenario_names[i]}.png")
        elif save_path:
            fname = save_path
            if i > 0:
                import os
                base, ext = os.path.splitext(save_path)
                fname = f"{base}_{i}{ext}"
            plt.savefig(fname)
            
        if show:
            plt.show()
        else:
            plt.close()

    # Stats comparison -------------------------------------------------------
    scenarios = {
        "Étoile seule": {
            "bright": (details["brights_so"], analytical_brights_so),
            "darks": (details["darks_so"], analytical_darks_so),
            "kernels": (data_so, analytical_data_so)
        },
        "Planète seule": {
            "bright": (details["brights_po"], analytical_brights_po),
            "darks": (details["darks_po"], analytical_darks_po),
            "kernels": (data_po, analytical_data_po)
        },
        "Complet": {
            "bright": (details["brights"], analytical_brights),
            "darks": (details["darks"], analytical_darks),
            "kernels": (data, analytical_data)
        }
    }

    if verbose:
        print("\n# Comparaison des Statistiques (analytique | numérique)")

        print("\n## Total Photons (Bright + Sum(Darks))")
        for name, data_dict in scenarios.items():
            num_b, ana_b = data_dict["bright"]
            num_d, ana_d = data_dict["darks"]
            total_num = np.mean(num_b) + np.sum(np.mean(num_d, axis=0))
            total_ana = np.mean(ana_b) + np.sum(np.mean(ana_d, axis=0))
            print(f"- {name}: {total_ana:.3e} | {total_num:.3e}")

        print("\n## Bright")
        for stat_name, stat_func in [("Moyenne", np.mean), ("Médiane", np.median), ("Std", np.std)]:
            print(f"\n### {stat_name}")
            for name, data_dict in scenarios.items():
                num, ana = data_dict["bright"]
                print(f"- {name}: {stat_func(ana):.3e} | {stat_func(num):.3e}")

        for k in range(6):
            print(f"\n## Dark {k+1}")
            for stat_name, stat_func in [("Moyenne", np.mean), ("Médiane", np.median), ("Std", np.std)]:
                print(f"\n### {stat_name}")
                for name, data_dict in scenarios.items():
                    num, ana = data_dict["darks"]
                    print(f"- {name}: {stat_func(ana[:, k]):.3e} | {stat_func(num[:, k]):.3e}")

        for k in range(3):
            print(f"\n## Kernel {k+1}")
            for stat_name, stat_func in [("Moyenne", np.mean), ("Médiane", np.median), ("Std", np.std)]:
                print(f"\n### {stat_name}")
                for name, data_dict in scenarios.items():
                    num, ana = data_dict["kernels"]
                    print(f"- {name}: {stat_func(ana[:, k]):.3e} | {stat_func(num[:, k]):.3e}")

    return {
        "numerical": details,
        "analytical": {
            "data": analytical_data,
            "data_so": analytical_data_so,
            "data_po": analytical_data_po,
            "brights": analytical_brights,
            "brights_so": analytical_brights_so,
            "brights_po": analytical_brights_po,
            "darks": analytical_darks,
            "darks_so": analytical_darks_so,
            "darks_po": analytical_darks_po
        }
    }

def time_evolution(ctx: Context=None, n=100, map=np.median, save_as=None, show=True) -> np.ndarray:
    """
    Get the time evolution of the kernel nuller.

    Parameters
    ----------
    ctx : Context
        The context to use.
    n : int, optional
        The number of samples to take at a given time, by default 1000.
    map : function, optional
        The function to use to map the data, by default np.median.
    save_as : str, optional
        Path or directory to save the figures.
    show : bool, optional
        Whether to show the plot. Default is True.

    Returns
    -------
    np.ndarray
        The time evolution of the kernel nuller. (n_h, 3)
    np.ndarray
        The reference time evolution of the kernel nuller (without input perturbation). (n_h, 3)
    """

    # Bluid context
    if ctx is None:
        ctx:Context = Context.get_VLTI()
        ctx.interferometer.chip.σ = np.zeros(14) * u.um
        ctx.target.companions[0].c = 0.1
    else:
        ctx = deepcopy(ctx)
        if ctx.target.companions == []:
            raise ValueError('No companion in the context. Please add a companion to the target.')
    ctx.Δh = ctx.interferometer.camera.e.to(u.hour).value * u.hourangle

    ctx_so = deepcopy(ctx)
    ctx_so.target.companions = []

    ctx_po = deepcopy(ctx)
    scale = 1e12
    ctx_po.target.f /= scale # Scale down the star flux to make it negligible
    ctx_po.target.companions[0].c *= scale

    # Numerical model ---------------------------------------------------------

    # Prepare data arrays
    data = np.empty((len(ctx.get_h_range()), 3))
    data_so = np.empty((len(ctx.get_h_range()), 3))
    data_po = np.empty((len(ctx.get_h_range()), 3))
    
    ref_data = np.empty((len(ctx.get_h_range()), 3))
    ref_data_so = np.empty((len(ctx.get_h_range()), 3))
    ref_data_po = np.empty((len(ctx.get_h_range()), 3))

    # Sample data
    for i, h in enumerate(ctx.get_h_range()):
        ctx.h = h
        ctx_so.h = h
        ctx_po.h = h

        # Generate noise
        upstream_pistons = np.random.normal(0, ctx.Γ.value, size=(n, len(ctx.interferometer.telescopes))) * ctx.Γ.unit

        # Distrib with companion(s)
        kernels = np.empty((n, 3))
        for j in range(n):
            outs = ctx.observe(upstream_pistons=upstream_pistons[j])
            kernels[j, :] = ctx.interferometer.chip.process_outputs(outs)
        data[i, :] = map(kernels, axis=0)
        
        # Distrib with star only
        kernels_so = np.empty((n, 3))
        for j in range(n):
            outs_so = ctx_so.observe(upstream_pistons=upstream_pistons[j])
            kernels_so[j, :] = ctx_so.interferometer.chip.process_outputs(outs_so)
        data_so[i, :] = map(kernels_so, axis=0)

        # Distrib with planet only
        kernels_po = np.empty((n, 3))
        for j in range(n):
            outs_po = ctx_po.observe(upstream_pistons=upstream_pistons[j])
            kernels_po[j, :] = ctx_po.interferometer.chip.process_outputs(outs_po)
        data_po[i, :] = map(kernels_po, axis=0)

        # Reference data
        outs = ctx.observe()
        ref_data[i, :] = ctx.interferometer.chip.process_outputs(outs)
        
        outs_so = ctx_so.observe()
        ref_data_so[i, :] = ctx_so.interferometer.chip.process_outputs(outs_so)

        outs_po = ctx_po.observe()
        ref_data_po[i, :] = ctx_po.interferometer.chip.process_outputs(outs_po)

    # Auto-save logic
    if save_as:
        utils.save_plot(save_as, "time_evolution.png")
    
    if show:
        plt.show()
    else:
        plt.close()

    return data, ref_data