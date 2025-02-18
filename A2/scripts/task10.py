from ase.io import read, write
from gpaw import GPAW, PW, restart
from ase.units import Bohr
import numpy as np

# Define the clusters
clusters = {
    'Na6': read('./A2/Na6_lowest_energy_structure.xyz'),
    'Na7': read('./A2/Na7_lowest_energy_structure.xyz'),
    'Na8': read('./A2/Na8_lowest_energy_structure.xyz')
}

# Perform GPAW calculations and save wavefunctions
for name, cluster in clusters.items():
    calc = GPAW(mode = PW(450), 
                xc='PBE', txt=f'task10_{name}_gpaw_output.log',
                nbands=28, 
                spinpol=False, 
                symmetry='off', 
                setups={'Na': '1'}
                )

    cluster.calc = calc  
    energy = cluster.get_potential_energy()
    calc.write(f'./A2/task10_{name}_wavefunctions.gpw', mode='all')  # Save all data including wavefunctions
    print(f'{name} energy: {energy} eV')

# Save wavefunctions to cube files
for name in clusters.keys():
    atoms, calc = restart(f'./A2/task10_{name}_wavefunctions.gpw')
    nbands = calc.get_number_of_bands()
    nelectrons = calc.get_number_of_electrons()
    occupied_bands = int(np.ceil(nelectrons / 2))
    print(f'{name} total bands: {nbands}')
    print(f'{name} occupied bands: {occupied_bands}')

    for band in range(occupied_bands):
        wf = calc.get_pseudo_wave_function(band=band)
        fname = f'./A2/task10_{name}_band_{band}.cube'
        print(f'writing wf {band} to file {fname}')
        write(fname, atoms, data=wf * Bohr**1.5)