from ase.db import connect
from ase.io import write
import numpy as np

def structures_are_different(struct1, struct2, threshold=0.5):
    """Compare two structures with a position threshold (in Angstroms)"""
    pos1 = struct1.toatoms().get_positions()
    pos2 = struct2.toatoms().get_positions()
    return np.abs(pos1 - pos2).min() > threshold

# Connect to database
db = connect('../gadb.db')

# Get sorted entries by energy
rows = db.select(sort='energy')
first_structure = next(rows)
second_structure = None

# Find second structure with different positions
for row in rows:
    if structures_are_different(first_structure, row, threshold=0.5):
        second_structure = row
        break

# Calculate energy difference
energy_diff = second_structure.energy - first_structure.energy
print(f"Lowest energy: {first_structure.energy:.6e} eV")
print(f"Second lowest energy: {second_structure.energy:.6e} eV")
print(f"Energy difference: {energy_diff:.6e} eV")

# Save structures
write('task7_lowest_energy_structure.xyz', first_structure.toatoms())
write('task7_second_lowest_energy_structure.xyz', second_structure.toatoms())