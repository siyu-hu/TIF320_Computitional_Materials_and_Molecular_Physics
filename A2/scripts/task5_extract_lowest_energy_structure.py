from ase.db import connect
from ase.io import write

# Connect to the database 
db = connect('gadb.db')

# Find the structure with the lowest energy
rows = db.select(sort='energy', limit=1)
lowest_energy_row = next(rows)
avg_atom_energy  =  lowest_energy_row.energy / 7 # ATTENTION: changing for Na8/Na7/Na6/
lowest_energy_structure = lowest_energy_row.toatoms()

print(f"Lowest energy: {avg_atom_energy:.6e} eV")
write('Na7_lowest_energy_structure.xyz', lowest_energy_structure)

