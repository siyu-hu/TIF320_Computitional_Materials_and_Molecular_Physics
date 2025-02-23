import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk
from gpaw import GPAW, PW

# Sodium bcc structure lattice constant a = 4.29 Å
atoms = bulk('Na', 'bcc', a=4.29)  

k_values = [(2, 2, 2), (4, 4, 4), (6, 6, 6), (8, 8, 8), (10, 10, 10)]
energies_k = []

for k in k_values:
    calc = GPAW(
        mode=PW(450),      # fixed cutoff energy
        xc='PBE',         
        setups={'Na': '1'},  # only considering 3s¹ electron
        kpts=k,
        txt=f'./A3/task1_Na_kpts_{k[0]}.out'
    )
    atoms.set_calculator(calc)
    energy = atoms.get_potential_energy()
    energies_k.append(energy)
    print(f'k-points: {k}, Energy: {energy:.6f} eV')


ecut_values = [200, 300, 350, 400,450, 500, 600, 650, 700, 750]
energies_cutoff = []

for ecut in ecut_values:
    calc = GPAW(
        mode=PW(ecut),
        xc='PBE',
        setups={'Na': '1'},
        kpts=(6, 6, 6),  # fixed k-points 
        txt=f'./A3/task1_Na_cutoff_{ecut}.out'
    )
    atoms.set_calculator(calc)
    energy = atoms.get_potential_energy()
    energies_cutoff.append(energy)
    print(f'Energy cutoff: {ecut} eV, Energy: {energy:.6f} eV')


plt.figure(figsize=(10, 4))

# k-points convergence
plt.subplot(1, 2, 1)
plt.plot([k[0] for k in k_values], energies_k, 'o-', label="Energy vs. k-points")
plt.xlabel('k-points grid size')
plt.ylabel('Energy (eV)')
plt.title('k-points Convergence')
plt.grid(True)
plt.legend()

# cutoff energy - convergence
plt.subplot(1, 2, 2)
plt.plot(ecut_values, energies_cutoff, 'o-', label="Energy vs. cutoff energy")
plt.xlabel('Energy Cutoff (eV)')
plt.ylabel('Energy (eV)')
plt.title('Cutoff Energy Convergence')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('./A3/task1_Na_convergence.png')


final_kpts = (6, 6, 6)  
final_ecut = 500         

calc = GPAW(
    mode=PW(final_ecut),
    xc='PBE',
    setups={'Na': '1'},
    kpts=final_kpts,
    txt='Na_final.out'
)

atoms.set_calculator(calc)
final_energy = atoms.get_potential_energy()
print(f'Final energy with k-points={final_kpts} and cutoff={final_ecut} eV: {final_energy:.6f} eV')
