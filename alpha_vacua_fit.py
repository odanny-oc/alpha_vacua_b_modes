import camb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize, differential_evolution
from camb import model, initialpower

matplotlib.use("Qt5Agg")

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
})


def get_bb_spectrum(alpha, r, return_ell=False):
    """Get B-mode spectrum for given alpha and r"""
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, tau=0.06)
    pars.InitPower.set_params(
        As=2.4e-9 * np.exp(2 * alpha), ns=0.96, r=r
    )
    pars.WantTensors = True
    pars.DoLensing = False
    pars.set_for_lmax(900, lens_potential_accuracy=0)
    
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, spectra=("total",), CMB_unit="muK")
    
    if return_ell:
        ell = np.arange(powers['total'].shape[0])
        return ell, powers['total'][:, 2]
    return powers['total'][:, 2]


def chi_squared(params, target_spectrum, ell_min=2, ell_max=500):
    """
    Chi-squared metric between spectra over a given l range.
    """
    alpha, r = params
    
    # Bounds check
    if alpha < 0 or alpha > 0.5 or r < 0.001 or r > 0.5:
        return 1e10
    
    try:
        spectrum = get_bb_spectrum(alpha, r)
        spectrum_renorm = np.sqrt(spectrum[ell_min:ell_max])
        target_renorm = np.sqrt(target_spectrum[ell_min:ell_max])
        
        # Chi-squared on log scale (since you're using loglog plots)
        chi2 = np.sum((np.log(spectrum_renorm) - np.log(target_renorm))**2)
        return chi2
    except:
        return 1e10


def find_matching_pairs(target_alpha=0.0, target_r=0.2, n_pairs=10, 
                       ell_min=2, ell_max=500, ):
    """
    Find (alpha, r) pairs that approximately match the target spectrum.
    """
    
    # Get target spectrum
    print(f"Computing target spectrum (α={target_alpha}, r={target_r})...")
    target_spectrum = get_bb_spectrum(target_alpha, target_r)
    
        # Use the analytical approximation: α ≈ 0.5 * ln(r_target / r)
        # DON'T refine - just evaluate along the analytical curve
    pairs = []
    r_values = np.linspace(0.07, 0.2, n_pairs + 2)[1:-1]  # Exclude endpoints
    
    for r in r_values:
        # Skip the exact target value
        if abs(r - target_r) < 0.01:
            continue
            
        if r < target_r:
            alpha = 0.5 * np.log(target_r / r)
        else:
            alpha = -0.5 * np.log(r / target_r)
        
        chi2 = chi_squared([alpha, r], target_spectrum, ell_min, ell_max)
        pairs.append((alpha, r, chi2))
        print(f"α={alpha:.4f}, r={r:.4f}, χ²={chi2:.6f}")
    
    
    return pairs, target_spectrum


def plot_results(pairs, target_spectrum, target_alpha=0.0, target_r=0.2):
    """Plot the fitted spectra"""
    ell = np.arange(len(target_spectrum))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Spectra
    ax1.loglog(ell, np.sqrt(target_spectrum), 'k-', linewidth=2, 
               label=f'Target: $\\alpha={target_alpha}$, $r={target_r}$')
    
    slicing = len(pairs)%5

    for i, (alpha, r, chi2) in enumerate(pairs[::slicing]): # Plots approx 5 pairs
        spectrum = get_bb_spectrum(alpha, r)
        ax1.loglog(ell, np.sqrt(spectrum), '--', alpha=0.7,
                   label=f'$\\alpha={alpha:.3f}$, $r={r:.3f}$ ($\\chi^2={chi2:.2e}$)')
    
    ax1.set_xlabel(r'$\ell$', fontsize=12)
    ax1.set_ylabel(r'$\sqrt{\ell(\ell+1)C_\ell^{BB}/(2\pi)}$ [$\mu$K]', fontsize=12)
    ax1.set_xlim(2, 500)
    ax1.legend(fontsize=9)
    ax1.set_title('B-mode Power Spectra')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter space
    alphas = [p[0] for p in pairs]
    rs = [p[1] for p in pairs]
    chi2s = [p[2] for p in pairs]
    
    scatter = ax2.scatter(rs, alphas, c=chi2s, s=100, cmap='viridis_r', 
                          edgecolors='black', linewidth=1)
    ax2.plot(target_r, target_alpha, 'r*', markersize=20, label='Target')
    
    # Analytical curve
    r_analytic = np.linspace(0.01, 0.3, 100)
    alpha_analytic = np.where(r_analytic < target_r,
                              0.5 * np.log(target_r / r_analytic),
                              -0.5 * np.log(r_analytic / target_r))
    ax2.plot(r_analytic, alpha_analytic, 'r--', alpha=0.5, 
             label=r'$\alpha = \frac{1}{2}\ln(r_0/r)$')
    
    ax2.set_xlabel(r'$r$', fontsize=12)
    ax2.set_ylabel(r'$\alpha$', fontsize=12)
    ax2.set_title('Parameter Space')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label(r'$\chi^2$', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('../presentation/alpha_fit.svg')
    # plt.show()


if __name__ == "__main__":
    # Find matching pairs using analytical approximation (no refinement!)
    print("="*60)
    print("Finding approximate matches along degeneracy curve")
    print("="*60)
    pairs, target = find_matching_pairs(
        target_alpha=0.0, 
        target_r=0.2,
        n_pairs=20,
        ell_min=2,
        ell_max=500,
    )
    
    print(np.array(pairs)[::2].shape)
    print(np.array(target).shape)
    plot_results(pairs, target, target_alpha=0.0, target_r=0.2)
    
    print("\n" + "="*60)
    print("APPROXIMATE MATCHES:")
    print("="*60)
    print(f"{'α':>8} {'r':>8} {'χ²':>12}")
    print("-"*30)
    for alpha, r, chi2 in pairs[:10]:
        print(f"{alpha:8.4f} {r:8.4f} {chi2:12.6e}")
