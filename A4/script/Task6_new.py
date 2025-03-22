import matplotlib.pyplot as plt
from gpaw import GPAW
from gpaw.dos import DOSCalculator
from gpaw import setup_paths
setup_paths.insert(0, '/Users/initial/remove/gpaw-setups-24.11.0')

vac_gpw_files = [
    "wulff_vac_9atoms.gpw",
    "wulff_vac_15atoms.gpw",
    "wulff_vac_27atoms.gpw",
    "wulff_vac_59atoms.gpw",
    "wulff_vac_65atoms.gpw"
]
plt.figure(figsize=(10, 12))

for i, file in enumerate(vac_gpw_files, 1):
    num_atoms = int(file.split('_')[2].replace('atoms.gpw', ''))

    calc = GPAW(file, txt=None)  
    dos = DOSCalculator.from_calculator(calc)
    energies = dos.get_energies()
    dos_values = dos.raw_dos(energies, width=0.1)

    plt.subplot(5, 1, i)
    plt.plot(energies, dos_values, label=f'{num_atoms} atoms (vacancy)')

    plt.xlabel(r'$\epsilon - \epsilon_F \ \rm{(eV)}$')
    plt.ylabel('Density of States (1/eV)')
    plt.title(f'DOS for {num_atoms} Na Nanoparticles (Vacancy)')
    plt.legend()
    plt.grid(True)

plt.tight_layout()

plt.savefig("dos_with_vacancy.png", dpi=300)

plt.show()