# Import necessary modules from ASE
from ase.build import bulk, surface
from ase.visualize import view

# Generate bulk sodium (Na) structure with bcc lattice
# Parameters:
#   'Na': Element symbol
#   'bcc': Crystal structure (body-centered cubic)
#   a=4.23: Lattice constant in Angstroms (experimental value for Na)
#   cubic=True: Force cubic unit cell
na_bulk = bulk('Na', 'bcc', a=4.23, cubic=True)

# Create (100) surface slab
# surface() parameters:
#   na_bulk: Bulk structure template
#   (1,0,0): Miller indices for (100) surface
#   layers=5: Number of atomic layers
na_100 = surface(na_bulk, (1, 0, 0), layers=5)
# Add 10 Angstrom vacuum layer along z-axis (axis=2)
na_100.center(vacuum=10, axis=2)

# Create (111) surface slab
na_111 = surface(na_bulk, (1, 1, 1), layers=5)
na_111.center(vacuum=10, axis=2)

# Create (110) surface slab
na_110 = surface(na_bulk, (1, 1, 0), layers=5)
na_110.center(vacuum=10, axis=2)

# Save structures to CIF files (optional)
na_100.write('./A3/task5_Na_100.cif')
na_111.write('./A3/task5_Na_111.cif')
na_110.write('./A3/task5_Na_110.cif')
