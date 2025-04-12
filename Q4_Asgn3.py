import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Planck 2018 cosmology parameters
c=3e5
H0=70
Omega_m = 0.315
Omega_Lambda = 0.685
Omega_r = 9.2e-5  # approximate radiation density parameter at z=0
Omega_k = 1.0 - Omega_m - Omega_Lambda - Omega_r  # curvature term (should be ~0)

def E_planck(z):
    """Generalized Friedmann equation with matter, radiation, curvature, and Lambda"""
    return np.sqrt(Omega_r * (1 + z)**4 + Omega_m * (1 + z)**3 + Omega_k * (1 + z)**2 + Omega_Lambda)

def comoving_distance_planck(z):
    integral, _ = quad(lambda z: 1 / E_planck(z), 0, z)
    return (c / H0)*integral  # in Mpc

def weinberg_parallax_distance_planck(z):
    chi = comoving_distance_planck(z)
    return chi  # in Mpc, assuming flat space

def singal_parallax_distance_planck(z):
    I = H0/c * comoving_distance_planck(z)
    return c/H0 *I/(1+I)

# Define the redshift range for evaluation, which was missing after the kernel reset
z_values = np.linspace(0.01, 5, 100)  # Avoid z=0 to prevent division by zero or singularities

# Recompute for Planck cosmology
weinberg_planck = [weinberg_parallax_distance_planck(z) for z in z_values]
singal_planck = [singal_parallax_distance_planck(z) for z in z_values]

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(z_values, weinberg_planck, label="Weinberg Parallax Distance", linestyle='--')
plt.plot(z_values, singal_planck, label="Singal Corrected Distance", linestyle='-')
plt.xlabel("Redshift z")
plt.ylabel("Parallax Distance (Mpc)")
plt.title("Parallax Distance vs Redshift (Planck 2018 Cosmology)")
plt.legend()
plt.tight_layout()
plt.savefig("Parallax Distances Comparision")
plt.show()

