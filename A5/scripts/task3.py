import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# Function to create the second derivative matrix using finite differences
def create_matrix_D2_finite_difference(N, h):
    """
    Creates the matrix for the second derivative using finite differences.
    Parameters:
        N (int): Number of grid points.
        h (float): Grid spacing.
    
    Returns:
        D2 (numpy array): Matrix for the second derivative.
    """
    D2 = np.zeros((N, N))
    i, j = np.indices(D2.shape)

    # Operator matrix for numerical second derivative
    # Formula: d2y/dx2 = (y(k-1) - 2y(k) + y(k+1)) / dx^2
    D2[i == j] = -2 / h**2
    D2[abs(i - j) == 1] = 1 / h**2

    return D2

# Normalize the radial wavefunction such that its total probability is 1
def normalize_radial_wavefunction(psi, r):
    """
    Normalize the radial wavefunction such that the total probability equals 1.
    Parameters:
        psi (numpy array): The radial wavefunction.
        r (numpy array): The radial grid.
    
    Returns:
        normalized_psi (numpy array): The normalized radial wavefunction.
    """
    norm = trapezoid(psi**2, r)  # Integrate psi^2 over r
    return psi / np.sqrt(norm)

# Compute the total probability of the radial wavefunction
def total_probability_of_radial_wavefunction(psi, r):
    """
    Compute the total probability of the radial wavefunction.
    Parameters:
        psi (numpy array): The radial wavefunction.
        r (numpy array): The radial grid.
    
    Returns:
        probability (float): The total probability, which should be 1 for a normalized wavefunction.
    """
    return trapezoid(psi**2, r)

# Solve Poisson's equation d2U/dr2 = -u2/r using finite differences
def solve_poisson(r, u):
    """
    Solves Poisson's equation in radial coordinates using finite differences.
    Parameters:
        r (numpy array): Radial grid points.
        u (numpy array): Source term for Poisson's equation.
    
    Returns:
        U (numpy array): Potential solution to Poisson's equation.
    """
    h = r[1] - r[0]
    N = len(r)

    # Create second derivative matrix
    D2 = create_matrix_D2_finite_difference(N, h)

    # Solve Poisson's equation with boundary conditions U(0) = 0, U(r_max) = 1
    U_0 = np.linalg.solve(D2, -u**2 / r)

    # Fix boundary conditions
    U = U_0 + r / np.max(r)

    return U

def solve_kohn_sham(r, potential):
    """
    Solves the radial Kohn-Sham equation for a given potential.
    Parameters:
        r (numpy array): Radial grid points.
        potential (numpy array): The potential for the Kohn-Sham equation.
    
    Returns:
        eps (float): The energy of the lowest state.
        u (numpy array): The corresponding wavefunction for the lowest energy.
    """
    h = r[1] - r[0]
    N = len(r)
    
    D2 = create_matrix_D2_finite_difference(N, h)
    potential_matrix = np.diag(potential)

    # Solve Kohn-Sham matrix equation
    eps_vec, u_mat = np.linalg.eigh(-0.5 * D2 + potential_matrix)

    # eigh will sort eigenvalues in ascending order
    eps = eps_vec[0] 
    u = u_mat[:,0]

    if u[0] < 0: u = -u
    norm2 = trapezoid(u**2, r)
    u /= np.sqrt(norm2)

    return eps, u

# Define the radial grid (excluding r=0 to avoid singularity)
N = 100
linspace_start, linspace_end = 0, 10
r = np.linspace(linspace_start, linspace_end, N+1)[1:]
h = r[1] - r[0]

# Hydrogen atom functions: Direct implementation in the main loop
a0 = 1  # Bohr radius [a.u]

# Ground state wavefunction of hydrogen
def ground_state_wavefunction(r, a0=1):
    return (1 / np.sqrt(np.pi)) * (1 / a0**(3/2)) * np.exp(-r / a0)

# Ground state energy of hydrogen (theoretically -0.5 [a.u])
def ground_state_energy():
    return -0.5

# Calculate electron density for hydrogen's ground state
electron_density = np.exp(-2 * r) / (np.pi * a0**3)  # 1/(pi * a0^3) in normalized form for hydrogen

# Solve Poisson's equation to obtain the potential
u = np.sqrt(4 * np.pi * electron_density) * r
U = solve_poisson(r, u)
V_sH = U / r  # Hartree potential

# Solve for hydrogen atom as a test case (Coulomb potential)
potential = -1 / r  # Potential for hydrogen atom
E_hydrogen, u_hydrogen = solve_kohn_sham(r, potential)

# Compute wavefunction from u
psi_hydrogen = u_hydrogen / (np.sqrt(4 * np.pi) * r)  # Normalize wavefunction in spherical coordinates
psi_hydrogen = normalize_radial_wavefunction(psi_hydrogen, r)

# Compare with theoretical wavefunction for hydrogen atom
psi_hydrogen_theoretical = ground_state_wavefunction(r)

# Print to verify reasonability
print("\nHydrogen")
print(f"Ground state energy: {E_hydrogen:.7f} (theoretically: {ground_state_energy():.7f} [a.u] )")
print(f"Wavefunction's total probability (theoretical): {total_probability_of_radial_wavefunction(psi_hydrogen_theoretical, r):.6f} (normalized if 1)")
print(f"Wavefunction's total probability (calculated): {total_probability_of_radial_wavefunction(psi_hydrogen, r):.6f} (normalized if 1)")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(r, psi_hydrogen_theoretical, color='blue', marker='^', linestyle='-', label='Theoretical hydrogen wavefunction')
plt.plot(r, psi_hydrogen, color='red', marker='.', linestyle='--', label='Calculated hydrogen wavefunction')
plt.xlabel('Radial distance r [a.u]')
plt.ylabel('Wavefunction')
plt.title('Hydrogen atom ground state wavefunction')
plt.grid()
plt.legend()
plt.savefig(f'./A5/task3_hydrogen_wavefunction_N={N}.png')

# Save data to CSV for further analysis or plotting
output_path_wavefunction = f'./A5/task3_hydrogen_wavefunction_N={N}.csv'  
with open(output_path_wavefunction, 'w') as CSV_file:
    CSV_file.write(f"Radial distance r [a.u], theoretical hydrogen ground state wavefunction , calculated hydrogen ground state wavefunction with {N} points\n")
    for line in range(N):
        CSV_file.write(f"{r[line]}, {psi_hydrogen_theoretical[line]}, {psi_hydrogen[line]}\n")

output_path_energy = f'./A5/task3_hydrogen_energy_N={N}.txt'  
with open(output_path_energy, 'w') as file:
    file.write(f"Calculated ground state energy of hydrogen: {E_hydrogen:.7f} [a.u]\n")
    file.write(f"Theoretical ground state energy of hydrogen: - 0.5 [a.u]\n")
    file.write(f"Number of points in discretized radial coordinate: {N}")
