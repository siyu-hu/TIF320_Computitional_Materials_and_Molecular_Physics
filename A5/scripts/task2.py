import numpy as np
import matplotlib.pyplot as plt

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
    # Grid spacing
    h = r[1] - r[0]
    N = len(r)

    # Create second derivative matrix
    D2 = create_matrix_D2_finite_difference(N, h)

    # Solve Poisson's equation with boundary conditions U(0) = 0, U(r_max) = 1
    U_0 = np.linalg.solve(D2, -u**2 / r)

    # Fix boundary conditions
    U = U_0 + r / np.max(r)

    return U

if __name__ == "__main__":

    # Define the radial grid
    N = 1000
    start, end = 0, 10
    r = np.linspace(start, end, N+1)[1:]  # Exclude r = 0 to avoid singularity
    h = r[1] - r[0]  # Grid spacing

    # Calculate the electron density for hydrogen's ground state
    electron_density = np.exp(-2 * r) / (np.pi)  # 1/(pi * a0^3) in normalized form for hydrogen

    # Solve Poisson's equation to obtain the potential
    u = np.sqrt(4 * np.pi * electron_density) * r
    U = solve_poisson(r, u)
    V_sH = U / r  # Hartree potential

    # Calculate the theoretical Hartree potential
    V_Hartree = 1.0 / r - (1.0 + 1.0 / r) * np.exp(-2.0 * r)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(r, V_Hartree, color='blue', marker='*', linestyle='', label='Theoretical Hartree potential')
    plt.plot(r, V_sH, color='red', marker='', linestyle='--', label='Calculated Hartree potential')
    plt.title('Hartree potential for hydrogen atom')
    plt.xlabel('Radial distance r (atomic units)')
    plt.ylabel('Energy of potential (atomic units)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'./A5/task2_Hartree_potential_N={N}.png')
    plt.show()

    # Save data to CSV for further analysis or plotting
    np.savetxt(f'./A5/task2_Hartree_potential_N={N}.csv', 
               np.column_stack([r, V_Hartree, V_sH]), 
               header="Radial distance (atomic units), Theoretical Hartree potential, Calculated Hartree potential", 
               delimiter=",", comments="")
