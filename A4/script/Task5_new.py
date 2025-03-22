from ase.io import read
from ase.atoms import Atoms
from gpaw import GPAW, PW, FermiDirac
import numpy as np
import glob
from ase.optimize import BFGS
from ase.neighborlist import NeighborList

relaxed_files = glob.glob('na_wulff_*_relaxed.traj')
vacancy_energies = {}

for file in relaxed_files:
    atoms = read(file)
    num_atoms = len(atoms)

    calc = GPAW(mode=PW(500),  
                xc='PBE',  
                kpts={'size': (1, 1, 1)},  
                occupations=FermiDirac(0.1),
                setups={'Na': '1'},
                txt=None)

    atoms.calc = calc
    E_wulff = atoms.get_potential_energy()  
    calc.write(f'wulff_{num_atoms}atoms.gpw')

    positions = atoms.get_positions()
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    center_atom_index = np.argmin(distances)

    atoms_v = atoms.copy()
    atoms_v.pop(center_atom_index)

    calc_v = GPAW(mode=PW(500),
                    xc='PBE',
                    kpts={'size': (1, 1, 1)},
                    occupations=FermiDirac(0.1),
                    setups={'Na': '1'},
                    txt=None)

    atoms_v.calc = calc_v
    opt = BFGS(atoms_v)
    opt.run(fmax=0.05)  
    E_wulff_v = atoms_v.get_potential_energy()
    calc_v.write(f'wulff_vac_{num_atoms}atoms.gpw')
    opt = BFGS(atoms_v)
    opt.run(fmax=0.05)  

    E_vacancy = E_wulff_v - E_wulff
    vacancy_energies[num_atoms] = E_vacancy

    print(f"Vacancy formation energy for {num_atoms} atoms: {E_vacancy:.4f} eV")

sorted_vacancy_energies = dict(sorted(vacancy_energies.items()))
print("\nVacancy formation energy (sorted by size):")
for size, energy in sorted_vacancy_energies.items():
    print(f"{size} atoms: {energy:.4f} eV")



