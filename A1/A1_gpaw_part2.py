import numpy as np
import matplotlib.pyplot as plt
import os
import time
from ase import Atoms
from gpaw import GPAW, PW
from ase.visualize import view

# Define output directory for results
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "results_2")
if not os.path.exists(output_dir):
    os.makedirs(output_dir) 

# Function to compute total energy given k-points or plane-wave cutoff
def compute_energy(param_list, param_type):
    results = []
    output_file = os.path.join(output_dir, f'2_results-{param_type}.txt')
    with open(output_file, 'w') as resultfile:
        for param in param_list:
            start_time = time.time()
            name = f'2_bulk-fcc-{param_type}-{param:.1f}'
            a = 4.05  # Experimental lattice constant
            b = a / 2
            
            bulk = Atoms('Al',
                         cell=[[0, b, b], [b, 0, b], [b, b, 0]],
                         pbc=True)
            
            if param_type == "k":
                calc = GPAW(mode=PW(350), kpts=(param, param, param), txt=os.path.join(output_dir, name + '.txt'))
            elif param_type == "pw":
                calc = GPAW(mode=PW(param), kpts=(8, 8, 8), txt=os.path.join(output_dir, name + '.txt'))

            bulk.calc = calc
            energy = bulk.get_potential_energy()
            end_time = time.time()
            elapsed_time = end_time - start_time

            results.append([param, energy, elapsed_time])
            print(param, energy, elapsed_time, file=resultfile)

    return np.array(results)

# Function to compute lattice constant optimization
def compute_lattice_constant(xc_type, structure):
    results = []
    output_file = os.path.join(output_dir, f'results-{structure}-{xc_type}.txt')
    with open(output_file, 'w') as resultfile:

        if structure == "fcc":
            a_values = np.linspace(3.9, 4.1, 10)  # 
        elif structure == "bcc":
            a_fcc = 4.033
            a_bcc = a_fcc * (2/4) ** (1/3) 
            print(f'a_bcc = ', a_bcc)
            a_values = np.linspace(a_bcc-0.1, a_bcc+0.1, 10)  # bcc lattice constant value is larger than fcc
    
        for a in a_values:
            start_time = time.time()
            name = f'2_bulk-{structure}-{xc_type}-{a:.3f}'
            b = a / 2
            
            if structure == "fcc":
                bulk = Atoms('Al',
                             cell=[[0, b, b], 
                                   [b, 0, b], 
                                   [b, b, 0]],
                             pbc=True)
            elif structure == "bcc":
                bulk = Atoms(['Al', 'Al'],
                             cell=[[a, 0, 0], 
                                   [0, a, 0], 
                                   [0, 0, a]], 
                             positions=[[0, 0, 0], [b, b, b]], 
                             pbc=True)                        
                #view(bulk)
            
            calc = GPAW(mode=PW(400), kpts=(8, 8, 8), xc=xc_type, txt=os.path.join(output_dir, name + '.txt'))
            bulk.calc = calc
            energy = bulk.get_potential_energy()
            end_time = time.time()
            elapsed_time = end_time - start_time

            results.append([a, energy, elapsed_time])
            print(a, energy, elapsed_time, file=resultfile)

    return np.array(results)

# Define k-points and plane-wave cutoffs
k_points = [4, 6, 8, 10, 12, 16, 18]
pw_cutoffs = [200, 250, 300, 350, 400, 450]

# Compute energy convergence
k_results = compute_energy(k_points, "k")
pw_results = compute_energy(pw_cutoffs, "pw")

print("Energy convergence results saved")


# Plot K-point convergence
fig, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot(k_results[:, 0], k_results[:, 1], 'b-o', label="Total Energy")
ax1.set_xlabel("K-point sampling")
ax1.set_ylabel("Total energy [eV]", color='b')

ax2 = ax1.twinx()
ax2.plot(k_results[:, 0], k_results[:, 2], 'r-s', label="Time")
ax2.set_ylabel("Time [s]", color='r')


plt.title("K-point Convergence")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "2_k_points_convergence.png"))
plt.close()

# Plot Plane-wave cutoff convergence
fig, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot(pw_results[:, 0], pw_results[:, 1], 'b-o', label="Total Energy")
ax1.set_xlabel("Plane-wave cutoff [eV]")
ax1.set_ylabel("Total energy [eV]", color='b')

ax2 = ax1.twinx()
ax2.plot(pw_results[:, 0], pw_results[:, 2], 'r-s', label="Time")
ax2.set_ylabel("Time [s]", color='r')

plt.title("Plane-wave Cutoff Convergence")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "2_pw_convergence.png"))
plt.close()

print("Energy convergence results figures(k-points and pw cutoff) saved")

#  7. * 8.
lda_fcc_results = compute_lattice_constant("LDA", "fcc")
pbe_fcc_results = compute_lattice_constant("PBE", "fcc")
lda_bcc_results = compute_lattice_constant("LDA", "bcc")
pbe_bcc_results = compute_lattice_constant("PBE", "bcc")

# Plot FCC
fig, ax1 = plt.subplots(figsize=(8, 6))

# FCC - LDA results
ax1.plot(lda_fcc_results[:, 0], lda_fcc_results[:, 1], 'b-o', label="LDA")
ax1.scatter(lda_fcc_results[:, 0], lda_fcc_results[:, 1], color='b')  
min_idx_lda = np.argmin(lda_fcc_results[:, 1])
min_x_lda, min_y_lda = lda_fcc_results[min_idx_lda, :2]   
ax1.annotate(f"({min_x_lda:.3f}, {min_y_lda:.3f})", (min_x_lda, min_y_lda), fontsize=10, ha='left', va='top', color='b')

#  FCC - PBE results
ax1.plot(pbe_fcc_results[:, 0], pbe_fcc_results[:, 1], 'r-s', label="PBE")
ax1.scatter(pbe_fcc_results[:, 0], pbe_fcc_results[:, 1], color='r') 
min_idx_pbe = np.argmin(pbe_fcc_results[:, 1]) 
min_x_pbe, min_y_pbe = pbe_fcc_results[min_idx_pbe, :2]
ax1.annotate(f"({min_x_pbe:.3f}, {min_y_pbe:.3f})", (min_x_pbe, min_y_pbe), fontsize=10, ha='left', va='top', color='r')

ax1.set_xlabel("Lattice constant [Å]")
ax1.set_ylabel("Total energy [eV]")
ax1.legend(loc='upper right')
plt.grid(True)
plt.title("Lattice Constant Optimization (fcc)")
plt.savefig(os.path.join(output_dir, "2_fcc_lattice_constant.png"))
plt.close()

print("FCC results figure saved")

# Plot BCC
fig, ax2 = plt.subplots(figsize=(8, 6))

# BCC - LDA results
ax2.plot(lda_bcc_results[:, 0], lda_bcc_results[:, 1], 'b-o', label="LDA")
ax2.scatter(lda_bcc_results[:, 0], lda_bcc_results[:, 1], color='b')  
min_idx_lda_bcc = np.argmin(lda_bcc_results[:, 1])  
min_x_lda_bcc, min_y_lda_bcc = lda_bcc_results[min_idx_lda_bcc, :2]
ax2.annotate(f"({min_x_lda_bcc:.3f}, {min_y_lda_bcc:.3f})", (min_x_lda_bcc, min_y_lda_bcc), fontsize=10, ha='left', va='top', color='b')

# BCC - PBE results
ax2.plot(pbe_bcc_results[:, 0], pbe_bcc_results[:, 1], 'r-s', label="PBE")
ax2.scatter(pbe_bcc_results[:, 0], pbe_bcc_results[:, 1], color='r')  
min_idx_pbe_bcc = np.argmin(pbe_bcc_results[:, 1])  
min_x_pbe_bcc, min_y_pbe_bcc = pbe_bcc_results[min_idx_pbe_bcc, :2]
ax2.annotate(f"({min_x_pbe_bcc:.3f}, {min_y_pbe_bcc:.3f})", (min_x_pbe_bcc, min_y_pbe_bcc), fontsize=10, ha='left', va='top', color='r')

ax2.set_xlabel("Lattice constant [Å]")
ax2.set_ylabel("Total energy [eV]")
ax2.legend(loc='upper right')
plt.grid(True)
plt.title("Lattice Constant Optimization (bcc)")
plt.savefig(os.path.join(output_dir, "2_bcc_lattice_constant.png"))
plt.close()
print("BCC results figure saved")