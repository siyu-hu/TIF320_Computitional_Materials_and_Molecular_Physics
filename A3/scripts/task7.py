import numpy as np
import matplotlib.pyplot as plt
from gpaw import GPAW, restart

surfaces = {
    "100": (1, 0, 0),
    "110": (1, 1, 0),
    "111": (1, 1, 1),
}
work_functions = {}

for name, miller in surfaces.items():
   
    atoms, calc = restart(f'slab-{miller}.gpw')
    
    e_fermi = calc.get_fermi_level()
    
    v = calc.get_electrostatic_potential().mean(axis=(0, 1))
    z = np.linspace(0, atoms.cell[2, 2], len(v))
    
    vacuum_start = int(len(z) * 0.85)
    vacuum_end = int(len(z) * 0.95)
    e_vacuum = np.median(v[vacuum_start:vacuum_end])
    
    work_functions[name] = e_vacuum - e_fermi

for name, wf in work_functions.items():
    print(f"Work Function ({name}): {wf:.3f} eV")