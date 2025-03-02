import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from task2 import solve_poisson
from task3 import solve_kohn_sham

def wavefunction_anzats(r):
    # constants from taks 1
    alpha  = np.array([0.297104, 1.236745, 5.749982, 38.216677])
    C      = np.array([-0.146876, -0.393152, -0.411198, -0.262007])

    psi = np.zeros_like(r)
    for p in range(0, len(C)):
        psi += C[p] * np.exp( - alpha[p] * r**2 )
    
    return psi

def n_s_from_u(r, u):
    return u**2 / (4 * np.pi * r**2)

def eps_exchange(n):
    return - (3/4) * (3 * n / np.pi)**(1/3)

def V_exchange(n):
    return - (3 * n / np.pi)**(1/3)


def eps_correlation(n):
    #constants from task description
    A = 0.0311
    B = -0.048
    C = 0.0020
    D = -0.0116
    gamma = -0.1423
    beta_1 = 1.0529
    beta_2 = 0.3334

    r_s = (3 / (4 * np.pi * n) )**(1/3)
    return np.where(r_s >=1, 
        gamma / (1 + beta_1*np.sqrt(r_s) + beta_2*r_s),
        A*np.log(r_s) + B + C*r_s*np.log(r_s) + D*r_s
    )

def V_correlation(n):
    #constants from task description
    A = 0.0311
    B = -0.048
    C = 0.0020
    D = -0.0116
    gamma = -0.1423
    beta_1 = 1.0529
    beta_2 = 0.3334

    r_s = (3 / (4 * np.pi * n) )**(1/3)
    return np.where(r_s >=1, 
        gamma * 
        (3 + 3.5 * beta_1 * np.sqrt(r_s) + 4 * beta_2 * r_s) /
        (3 * (1 + beta_1 * np.sqrt(r_s) + beta_2 * r_s))
        ,
        A * (np.log(r_s) - 1/3) + 
        B + 
        C * (2 * r_s * np.log(r_s) - r_s) / 3 + 
        2 * D * r_s / 3
    )

def find_scf_wavefunction(rmax, N, include_exchange, include_correlation):
    r = np.linspace(0, rmax, N+1)[1:]
    u = np.sqrt(4 * np.pi) * r * wavefunction_anzats(r)

    E = 0
    E_old = 0
    energy_list = []  # Store the energy at each iteration
    for i in range(100):
        U = solve_poisson(r, u)
        V_sH = U / r
        
        if include_exchange:
            V_H = 2 * V_sH
        else:
            V_H = V_sH

        n = 2 * n_s_from_u(r, u)

        eps_xc = np.zeros_like(r)
        V_xc = np.zeros_like(r)
        if include_exchange:
            eps_xc += eps_exchange(n)
            V_xc += V_exchange(n)
        if include_correlation:
            eps_xc += eps_correlation(n)
            V_xc += V_correlation(n)

        potential = - 2.0 / r + V_H + V_xc
        eps, u = solve_kohn_sham(r, potential)

        E_old = E
        E = 2 * eps - 2 * trapezoid(u**2 * (0.5 * V_H + V_xc - eps_xc), r)
        print(f"Iteration {i}: Energy = {E:.6f} [a.u.]")
        energy_list.append(E)

        if abs(E - E_old) < 1e-5:
            break
        
    psi = u / (np.sqrt(4 * np.pi) * r)
    print(f"Converged in {i} iterations")
    return E, r, psi, energy_list

# task4 DFT self-consistent field without exchange and correlation
# only the Hartree potential is included
# task = 4
# include_exchange = False
# include_correlation = False

# task5 DFT self-consistent field with exchange but without correlation
# task = 5
# include_exchange = True
# include_correlation = False

# task6 DFT self-consistent field with correlation but without exchange
# task = 6
# include_exchange = False
# include_correlation = True

# task7 DFT self-consistent field with exchange and correlation
task = 7
include_exchange = True
include_correlation = True

# E, r, psi = find_scf_wavefunction(30, 6000, include_exchange, include_correlation)
# print(f"Ground state energy: {E:.6f} (a.u.)")

# plt.plot(r, psi, color='black', marker='', linestyle='-', label='Computed helium wavefunction')
# plt.xlabel('Radial distance r (atomic units)')
# plt.ylabel('Wavefunction')
# plt.grid()
# plt.legend()
# plt.show()
a0 = 1
E, r, psi, energy_list = find_scf_wavefunction(30, 6000, include_exchange, include_correlation)

# Theoretical Hydrogen atom wavefunction  for comparison
psi_theoretical = (1 / np.sqrt(np.pi)) * (1 / a0**(3/2)) * np.exp(-r / a0)
print(f"Task{task}: Ground state energy: {E:.6f} [a.u.]")

# Plot energy convergence over iterations
plt.figure(figsize=(10, 6))
plt.plot(range(len(energy_list)), energy_list, color='blue', marker='o', linestyle='-', label="Energy convergence")
plt.xlabel('Iteration')
plt.ylabel('Energy [a.u.]')
plt.title(f'Task{task}:Energy Convergence in SCF Iterations with exchange= {include_exchange}, correlation = {include_correlation}')
plt.grid(True)
plt.legend()
plt.savefig(f"./A5/task{task}_energy_convergence_plot.png")  

# Plot the computed wavefunction
plt.figure(figsize=(10, 6))
plt.plot(r, psi_theoretical, color='blue', marker='^', linestyle='', label='Theoretical helium wavefunction')
plt.plot(r, psi, color='red', marker='', linestyle='-', label='Computed helium wavefunction')
plt.xlabel('Radial distance r [a.u.]')
plt.ylabel('Wavefunction')
plt.title(f'Task{task}: Helium atom ground state wavefunction with exchange= {include_exchange}, correlation = {include_correlation}')
plt.grid()
plt.legend()
plt.savefig(f'./A5/task{task}_helium_wavefunction_exchange={include_exchange}_correlation={include_correlation}.png')


with open(f'./A5/task{task}_helium_energy.txt', 'w') as file:
    file.write(f"Calculated ground state energy of helium: {E:.8f} [a.u.]\n")
    file.write(f"Number of points in discretized radial coordinate: 6000 \n")
    file.write(f"Exchange included: {include_exchange}\n")
    file.write(f"Correlation included: {include_correlation}\n")
    file.write(f"Wavefunction's total probability: {trapezoid(psi**2 * 4 * np.pi * r**2, r):.6f} (normalized if 1)\n")
