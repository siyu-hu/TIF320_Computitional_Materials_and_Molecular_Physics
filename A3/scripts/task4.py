from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from ase import Atoms as ASEAtoms
from ase.build import bulk
from gpaw import GPAW, PW
import time
import numpy as np
import matplotlib.pyplot as plt

print("\n=== Starting Phonon Calculation for BCC Na ===")
start_time = time.time()

# initialize BCC Na unit cell
na = bulk('Na', 'bcc', a=4.1932)
print("[1/6] Original BCC unit cell created (2 atoms).")

calc = GPAW(
    mode=PW(300),
    xc='PBE',
    kpts=(4, 4, 4),
    random=True,
    setups={'Na': '1'},
    symmetry='off',  
    txt='./A3/task4_Na_phonopy.txt'
)
print("[2/6] GPAW calculator configured (kpts=4x4x4, cutoff=600 eV).")

# initialize phonopy object
phonon = Phonopy(
    PhonopyAtoms(
        symbols=na.symbols,
        positions=na.positions,
        cell=na.cell
    ),
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    is_symmetry=False
)
print("[3/6] 2x2x2 supercell created (16 atoms).")

# Generate displacement configurations
phonon.generate_displacements(distance=0.01)
supercells = phonon.supercells_with_displacements
n_displacements = len(supercells)
print(f"[4/6] Generated {n_displacements} displacement configurations.")

# calculate forces
print(f"\n--- Calculating Forces ({n_displacements} configurations) ---")
force_list = []
for i, scell_phonopy in enumerate(supercells, 1):
    scell_ase = ASEAtoms(
        symbols=scell_phonopy.symbols,
        positions=scell_phonopy.positions,
        cell=scell_phonopy.cell,
        pbc=True
    )
    scell_ase.calc = calc
    print(f"  Processing displacement {i}/{n_displacements}...")
    forces = scell_ase.get_forces()
    force_list.append(forces)

phonon.forces = force_list
print("\n[5/6] All forces calculated. Building force constants...")
phonon.produce_force_constants()

# define the path for band structure calculation
path = [
    ("G", [0.0, 0.0, 0.0]),
    ("H", [0.0, 0.0, 1.0]),
    ("P", [0.5, 0.5, 0.5]),
    ("N", [0.5, 0.5, 0.0]),
    ("G", [0.0, 0.0, 0.0])
]

print("[6/6] Computing phonon band structure along G-H-P-N-G...")
phonon.run_band_structure(
    paths=[[path[0][1], path[1][1]],
          [path[1][1], path[2][1]],
          [path[2][1], path[3][1]],
          [path[3][1], path[4][1]]],
    labels=[point[0] for point in path]
)

#  Check if the phonon spectrum is stable
# frequencies = phonon.get_band_structure_dict()['frequencies']
# if np.any(frequencies < -1e-4): 
#     print("Warning: Imaginary frequencies detected!")
# else:
#     print("Phonon spectrum is dynamically stable.")
fig = phonon.plot_band_structure()
ax = fig.gca()
ax.set_title("Phonon Band Structure of BCC Sodium", fontsize=14, pad=15)
ax.set_xlabel("Wave Vector", fontsize=12)
ax.set_ylabel("Frequency (THz)", fontsize=12)
ax.tick_params(axis="both", labelsize=10)
ax.grid(axis="y", linestyle=":", alpha=0.5, color="gray")

plt.tight_layout()

fig.savefig(
    "./A3/task4_Na_phonon_bandstructure_pro.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)