from ase import Atoms
from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac
import numpy as np

#generate supercell
supercell_sizes = [2, 3, 4]
for supercell_size in supercell_sizes:
    Na_bulk = bulk('Na', 'bcc', a=4.1932)
    atoms = Na_bulk* [supercell_size, supercell_size, supercell_size]
    num_atoms = len(atoms)

    calc = GPAW(mode=PW(500),  
                xc='PBE',
                kpts=(4,4,4),  
                occupations=FermiDirac(0.05),
                txt=None)

    atoms.calc = calc
    E_bulk = atoms.get_potential_energy()

    center = atoms.get_center_of_mass()
    distances = np.linalg.norm(atoms.positions - center, axis=1)
    vacancy_index = np.argmin(distances)
    
    atoms_vacancy = atoms.copy()
    del atoms_vacancy[vacancy_index]

    calc_vac = GPAW(mode=PW(500),
                    xc='PBE',
                    kpts=(4,4,4),
                    occupations=FermiDirac(0.05),
                    txt=None)

    atoms_vacancy.calc = calc_vac
    E_vacancy = atoms_vacancy.get_potential_energy() 

    E_vac_form = E_vacancy - E_bulk
    print(f"{num_atoms} vacancy formation energy: {E_vac_form:.4f} eV")


