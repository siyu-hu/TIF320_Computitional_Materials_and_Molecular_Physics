# Import necessary modules from ASE
import os
from ase.build import bulk, surface
from ase.visualize import view

os.makedirs("A3", exist_ok=True)

na_bulk = bulk('Na', 'bcc', a=4.1932, cubic=True)

surfaces = {
    "100": (1, 0, 0, 10),
    "110": (1, 1, 0, 11),
    "111": (1, 1, 1, 11),
}

for name, (h, k, l, layers) in surfaces.items():
    slab = surface(na_bulk, (h, k, l), layers=layers)
    slab.center(vacuum=25, axis=2)  # add vacuum to the top and bottom of the slab
    filename = f"A3/task5_Na_{name}.cif"
    slab.write(filename) 
    print(f"Saved {filename}")


