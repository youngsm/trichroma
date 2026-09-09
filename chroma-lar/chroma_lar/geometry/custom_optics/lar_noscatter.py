import numpy as np
from chroma.geometry import Material
from .utils import make_prop

lar_noscatter = Material("liquid_argon_noscatter")

import math
import numpy as np

# ALL TAKEN FROM https://lar.bnl.gov/properties/!

def compute_density(T=87.0, T_c=150.687, rho_c=0.5356):
    """
    Computes the density of a fluid given its temperature and critical properties.

    Args:
        T (float): Temperature in Kelvin.
        T_c (float): Critical temperature in Kelvin. Default is for liquid argon.
        rho_c (float): Critical density in g/cm^3. Default is for liquid argon.

    Returns:
        float: Density in g/cm^3
    """
    # Coefficients from the empirical formula
    a = 1.5004262
    b = -0.31381290
    c = 0.086461622
    d = -0.041477525

    # Compute reduced temperature difference
    t = 1 - T / T_c

    # Logarithmic density expression
    log_rho = (
        math.log(rho_c)
        + a * math.pow(t, 0.334)
        + b * math.pow(t, 2.0 / 3.0)
        + c * math.pow(t, 7.0 / 3.0)
        + d * math.pow(t, 4.0)
    )

    # Return the exponentiated density
    return math.exp(log_rho)


lar_noscatter.set("density", compute_density(T=87.0))  # g/cm^3 at 87 K

# --- Refractive index (Seidel et al., NIM A 2002) --------------
def iof_l(lambda_nm, T=87.0, T_c=150.687, rho_c=0.5356):
    """
    Computes the liquid index of refraction for a given wavelength and temperature.

    Args:
        lambda_nm (float): Wavelength in nanometers.
        T (float): Temperature in Kelvin.
        T_c (float): Critical temperature in Kelvin (default: for liquid argon).
        rho_c (float): Critical density in g/cm^3 (default: for liquid argon).

    Returns:
        float: Liquid index of refraction.
    """

    # get gas iof
    lam_um = lambda_nm / 1000.0  # convert to microns

    c0 = 1.2055e-2
    a1, b1 = 0.2075, 91.012
    a2, b2 = 0.0415, 87.892
    a3, b3 = 4.3330, 214.02

    inv_lam2 = 1.0 / (lam_um**2)
    nG = 1 + c0 * (a1 / (b1 - inv_lam2) + a2 / (b2 - inv_lam2) + a3 / (b3 - inv_lam2))

    rhoL = compute_density(T)

    # get liquid iof
    rhoG = 1.0034 * 0.0017840
    num = np.sqrt((2 + nG**2) * rhoG + 2 * (-1 + nG**2) * rhoL)
    den = np.sqrt((2 + nG**2) * rhoG + rhoL - nG**2 * rhoL)

    return num / den

wvl = np.linspace(120, 300, 15)
n = iof_l(wvl)
# lar_noscatter.refractive_index = make_prop(wvl, n)
lar_noscatter.set("refractive_index", 1.233) # matching simlar

# --- Rayleigh scattering length (ArDM & DUNE numbers) ----------
wvl = np.array([128])
rayleigh = np.array([1e10])  # mm
lar_noscatter.scattering_length = make_prop(wvl, rayleigh)

# --- Absorption (choose ~30 m flat unless you have impurity model)-
lar_noscatter.absorption_length = make_prop([118.0, 800.0], [1e10, 1e10])  # mm

# --- Scintillation spectrum -------------------------------------

# don't worry for our purposes
# scint_wvl = np.linspace(118, 138, 101)
# scint_pdf = np.exp(
#     -((scint_wvl - 128.0) ** 2) / (2 * 4.2**2)
# )  # σ ≈ 4.2 nm (10 nm FWHM)
# scint_pdf /= np.trapz(scint_pdf, scint_wvl)  # normalise to 1
# lar_noscatter.scintillation_spectrum = make_prop(scint_wvl, scint_pdf)

# --- Time profile (singlet / triplet) ---------------------------
# don't worry for our purposes
# lar_noscatter.scintillation_waveform = {
#     "": make_prop([-6.0, -1600.0], [0.3, 0.7])  # τ in ns , relative weights
# }

lar_noscatter.scintillation_light_yield = 20000  # photons / MeV (@ 500V/cm)

# Birks/recombination quenching left at default (unity) ----------------
# also don't worry for our purposes


# from simlar
lar_noscatter.composition = {"Ar": 1.0}


__exports__ = ["lar_noscatter"]