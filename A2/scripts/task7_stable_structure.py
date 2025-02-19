from ase.db import connect
from ase.io import write
import numpy as np
import matplotlib.pyplot as plt


db = connect('./A2/na8_gadb.db') # ATTENTION: changing for Na8/Na7/Na6/
energies = []
indices = []
rows = db.select()

for idx, row in enumerate(rows):
    atoms = row.toatoms()
    if atoms.calc is not None:
        energies.append(atoms.get_total_energy())
    else:
        energies.append(np.nan)
    indices.append(idx)

valid_indices = [idx for idx, e in zip(indices, energies) if not np.isnan(e)]
valid_energies = [e for e in energies if not np.isnan(e)]

if valid_energies:  
    min_energy = min(valid_energies)
    min_index = valid_indices[valid_energies.index(min_energy)]
    print(f"Lowest energy: {min_energy:.6e} eV")
else:
    min_energy, min_index = None, None

plt.figure(figsize=(10, 6))
sc = plt.scatter(indices, energies, c=energies, cmap='viridis', edgecolors='k', s=50) 
plt.colorbar(label="Energy (eV)")

if min_energy is not None:
    plt.scatter(min_index, min_energy, color='red', s=100, marker='*', label=f"Lowest Energy ({min_energy:.4f} eV)")
    plt.legend()


for i, (x, y) in enumerate(zip(indices, energies)):
    plt.text(x, y, str(i), fontsize=8, ha='right', va='bottom')


plt.xlabel("Structure Index")
plt.ylabel("Energy (eV)")
plt.title("Task7 Na8 Energy Distribution of Structures") # ATTENTION: changing for Na8/Na7/Na6/
plt.legend()
plt.grid(True)
plt.savefig('./A2/task7_na8_energy_distribution.png') # ATTENTION: changing for Na8/Na7/Na6/
plt.show()

id_second_lowest = 75
print(f"length of energies: {len(energies)}")

second_lowest_energy = energies[id_second_lowest]  # eV
print(f"Second lowest energy: {second_lowest_energy:.6e} eV")
second_lowest_structure = db.get_atoms(id=id_second_lowest)
write(f'./A2/task7_Na8_structure_{id_second_lowest}.xyz', second_lowest_structure) # ATTENTION: changing for Na8/Na7/Na6/
