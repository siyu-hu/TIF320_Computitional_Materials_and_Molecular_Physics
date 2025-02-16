from ase import Atoms
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS
from gpaw import GPAW, FermiDirac

positions = np.array([
    [0.0, 0.0, 0.0],
    [2.5, 0.0, 0.0],
    [0.0, 2.5, 0.0],
    [2.5, 2.5, 0.0],
    [1.25, 1.25, 2.0],
    [1.25, 2.0, 1.25]
])
atoms = Atoms('Na6', positions=positions)

atoms.center(vacuum=5.0) # 5 Å vacuum in x, y, and z directions


calc = GPAW(
    mode='lcao',              # linear combination of atomic orbitals
    basis='dzp',              # double-zeta polarized basis set
    xc='PBE',                 # planewave basis set
    occupations=FermiDirac(0.1),
    txt='calc.log'            
)

atoms.calc = calc

# relax the structure
dyn = BFGS(atoms, logfile='opt.log')
dyn.run(fmax=0.05, steps=200)  # terminate when the forces are less than 0.05 eV/Å or 200 steps

# final energy
energy = atoms.get_potential_energy()
print(f"Final energy (eV) = {energy:.6f}")

calc.write('final_guess.gpw', mode='all')

write('final_guess.traj', atoms)
