import numpy as np
import matplotlib.pyplot as plt


alpha  = np.array([0.297104, 1.236745, 5.749982, 38.216677])

# Create Hamiltonian matrix
n = len(alpha)
h = np.zeros((n, n))
for p in range(n):
    for q in range(n):
        alpha_sum = alpha[p] + alpha[q]
        h[p][q] = 3 * np.pi * alpha[q] * np.sqrt(np.pi / alpha_sum**3) - \
                    3 * np.pi * alpha[q]**2 * np.sqrt(np.pi / alpha_sum**5) - \
                    4 * np.pi / alpha_sum

# Create overlap matrix
S = np.zeros((n, n))
for p in range(n):
    for q in range(n):
        S[p][q] = (np.pi / (alpha[p] + alpha[q]))**1.5

# Create Coulomb integral tensor
Q = np.zeros((n, n, n, n))
for p in range(n):
    for r in range(n):
        for q in range(n):
            for s in range(n):
                Q[p][r][q][s] = 2 * np.pi**2.5 / (
                    (alpha[p] + alpha[q]) * (alpha[r] + alpha[s]) *
                    np.sqrt(alpha[p] + alpha[q] + alpha[r] + alpha[s])
                )

# Pick initial value:
C = np.array([1, 1, 1, 1], dtype=np.float64)  # Ensure C is a float array

# Normalize coefficients using the overlap matrix
norm2 = 0
for p in range(n):
    for q in range(n):
        norm2 += C[p] * S[p][q] * C[q]
C /= np.sqrt(norm2)

# Compute initial energy
energy = 0
for p in range(n):
    for q in range(n):
        energy += 2 * C[p] * C[q] * h[p][q]
        for r in range(n):
            for s in range(n):
                energy += Q[p][r][q][s] * C[p] * C[q] * C[r] * C[s]
old_energy = energy

# Iterate to find self-consistent solution
for i in range(1000):
    # Create Fock matrix
    F = np.zeros((n, n))
    for p in range(n):
        for q in range(n):
            F[p][q] = h[p][q]
            for r in range(n):
                for s in range(n):
                    F[p][q] += Q[p][r][q][s] * C[r] * C[s]

    # Diagonalize overlap matrix to solve the generalized eigenvalue problem
    d, U = np.linalg.eigh(S)
    d[d < 1e-12] = 1e-12
    V = U @ np.diag(1 / np.sqrt(d))

    # Transform Fock matrix to new basis
    F_prime = V.T @ (F @ V)
    Eprime, Cprime = np.linalg.eigh(F_prime)

    # Transform back to original basis
    C = V @ Cprime[:, 0]

    # Normalize coefficients again
    norm2 = 0
    for p in range(n):
        for q in range(n):
            norm2 += C[p] * S[p][q] * C[q]
    C /= np.sqrt(norm2)

    # Compute new energy
    energy = 0
    for p in range(n):
        for q in range(n):
            energy += 2 * C[p] * C[q] * h[p][q]
            for r in range(n):
                for s in range(n):
                    energy += Q[p][r][q][s] * C[p] * C[q] * C[r] * C[s]

    # convergence criterion
    if (27.2114 * abs(energy - old_energy) < 1e-5):
        break

    old_energy = energy

print(f"Ground state energy: {energy:.7f} [Ha](ideally, it should be -2.8551716[Ha])")
print(f"C-parameters: {C}")

# Compute and save wavefunction
N = 1000
r_lin = np.linspace(0, 5, N)
phi = np.abs(np.zeros_like(r_lin))

for p in range(n):
    phi += C[p] * np.exp(-alpha[p] * r_lin**2)

# Save alpha parameters and C-parameters, with clear titles and alignment
with open('./A5/task1_helium_wavefunction.txt', 'w') as file:
    file.write("---- Calculated C-parameters for the helium ground state wavefunction ----\n")
    file.write("This file contains the C-parameters for the helium wavefunction based on four Gaussians as a basis, each on the form: C * exp(-alpha * r^2)\n\n")
    
    file.write("---- Alpha Parameters ----\n")
    for i in range(n):
        file.write(f"alpha[{i}] = {alpha[i]:.6f}\n") 

    file.write("\n---- C Parameters (Coefficients) ----\n")
    for i in range(n):
        file.write(f"C[{i}] = {C[i]:.6f}\n")
    
    file.write("\n----  RESULTS ----\n")
    file.write(f"Number of points in discretized radial coordinate: {N}\n")
    file.write(f"Ground state energy of Helium : {energy:.7f} [Ha]\n" )  
