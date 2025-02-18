# task10 by @jingyi zhou

from ase.io import read
from gpaw import GPAW, PW
import numpy as np

atoms_na7 = read('./A2/Na7_lowest_energy_structure.xyz')

calc = GPAW(mode = PW(450), xc='PBE', txt='test_Na7_wavefunc.log',nbands=28, spinpol=False, symmetry='off', setups={'Na': '1'})

atoms_na7.calc =calc
energy = atoms_na7.get_potential_energy()

n_electrons = calc.get_number_of_electrons()
n_occupied = int(np.ceil(n_electrons / 2)) 
print(f"n_electrons: {n_electrons}")
print(f"n_occupied:{n_occupied}") 
for band in range(n_occupied):
    wave = calc.get_pseudo_wave_function(band=band, spin=0, kpt=0)
    from ase.io.cube import write_cube
    with open(f'test_Na7_band{band}.cube', 'w') as f:
        write_cube(f, atoms_na7, data=wave)