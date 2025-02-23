from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac
from gpaw.dos import DOSCalculator
import matplotlib.pyplot as plt

# Create the sodium BCC structure
na = bulk('Na', 'bcc', a=4.1932)  # Lattice constant from task 2

# Set up the DFT calculation
calc = GPAW(mode=PW(600),                # Plane-wave cutoff from task 1
            xc='PBE',                   
            kpts=(8, 8, 8),           # k-point grid, from task 1
            random=True,                 # Random initial guess for wavefunctions
            setups={'Na': '1'},          # Only considering 3s¹ electron
            occupations=FermiDirac(0.01),# Smearing for metallic systems
            txt='./A3/task3_Na_gs.txt')        

# Attach calculator and run SCF calculation
na.calc = calc
na.get_potential_energy()
ef = calc.get_fermi_level()              
calc.write('./A3/task3_Na_gs.gpw')                 # Save the ground state results

# Band structure calculation
calc = GPAW('./A3/task3_Na_gs.gpw').fixed_density(
    nbands=16,                           # Number of bands to compute
    symmetry='off',                   
    kpts={'path': 'GHPNGH', 'npoints': 60},  # High-symmetry path for BCC sodium
    convergence={'bands': 8})            # Convergence settings


bs = calc.band_structure()
bs.plot(filename='./A3/task3_Na_bandstructure.png', show=False, emax=10.0)

# Total and Projected DOS calculation
dos = DOSCalculator.from_calculator('./A3/task3_Na_gs.gpw')
energies = dos.get_energies()
width = 0.1  # Smearing width

plt.figure(figsize=(10, 6))
if dos.nspins == 2: # Spin-polarized calculation
    plt.plot(energies - ef, dos.raw_dos(energies, spin=0, width=width))
    plt.plot(energies - ef, dos.raw_dos(energies, spin=1, width=width))
    plt.legend(('up', 'down'), loc='upper left')
    print('Spin-polarized calculation')
else:
    plt.plot(energies - ef, dos.raw_dos(energies, width=width))
    print('Non-spin-polarized calculation')


plt.xlabel(r'$\epsilon - \epsilon_F \ \rm{(eV)}$')
plt.ylabel('Density of States (1/eV)')
plt.title('Total Density of States of Na')
plt.axvline(0, color='gray', linestyle='--', label='Fermi level')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('./A3/task3_Na_total_DOS.png')
plt.show()


