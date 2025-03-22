from ase.cluster import wulff_construction
from ase.io import write

surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
esurf = [0.2098, 0.1987, 0.2439]
lc = 4.1932
sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for size in sizes:
    atoms = wulff_construction('Na', surfaces, esurf, size, 'bcc', rounding = 'below', latticeconstant=lc)
    print(f"The number of atoms: {len(atoms)}")