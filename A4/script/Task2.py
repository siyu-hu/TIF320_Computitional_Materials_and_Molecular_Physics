from ase.io import read, write
from ase.neighborlist import NeighborList
import numpy as np
import glob 
from gpaw import GPAW, PW, FermiDirac
from ase.optimize import LBFGS
from ase.cluster import wulff_construction

surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
esurf = [0.2098, 0.1987, 0.2439]
lc = 4.1932
sizes = [10, 20, 30, 60, 100]
for size in sizes:
    atoms = wulff_construction('Na', surfaces, esurf, size, 'bcc', rounding = 'below', latticeconstant=lc)
    filename = f"na_wulff_{len(atoms)}atoms.traj"
    write(filename, atoms)
    print(f"The number of atoms: {len(atoms)}")

input_files = glob.glob('na_wulff_*atoms.traj')

for file in input_files:
    atoms = read(file)
    if not atoms.cell.any():  
        atoms.center(vacuum=10.0)
        atoms.pbc = [False, False, False]  
    num_atoms = len(atoms)

    calc = GPAW(mode=PW(500),
               xc='PBE',
               kpts={'size': (1,1,1)}, 
               txt=f'relax_{num_atoms}atoms.log',
               parallel={'domain': 64},
               setups={'Na': '1'},
               h=0.18, occupations = FermiDirac(0.1), maxiter=300)
    
    atoms.calc = calc

    opt = LBFGS(atoms, trajectory=f'opt_{num_atoms}atoms.traj')
    opt.run(fmax=0.02)

    output_file = file.replace('.traj', '_relaxed.traj')
    write(output_file, atoms)
    print(f"Relaxed structure saved to: {output_file}")

    def calculate_avg_neighbor_distance(filename):
        atoms = read(filename)
        positions = atoms.get_positions()
        num_atoms = len(atoms)

        cutoff = 4.1932 * np.sqrt(3)/2 * 1.2
        nl = NeighborList([cutoff/2] * num_atoms, self_interaction=False, bothways=False)
        nl.update(atoms)

        nn_distances = []
 
        for i in range(num_atoms):
            indices, _ = nl.get_neighbors(i)
            for j in indices:
                if j != i:
                    d = np.linalg.norm(positions[i] - positions[j])
                    nn_distances.append(d)
        return {
            'mean': np.mean(nn_distances),
            'std': np.std(nn_distances),
            'min': np.min(nn_distances),
            'max': np.max(nn_distances)
        }

results = []
relaxed_files = glob.glob('na_wulff_*_relaxed.traj')

print("\n=== Average Nearest-Neighbor Distance ===")
for file in relaxed_files:
    stats = calculate_avg_neighbor_distance(file)
    num_atoms = len(read(file))
    results.append((num_atoms, stats))
    
    print(f"N={num_atoms:<4} | "
          f"Mean: {stats['mean']:.3f} ± {stats['std']:.3f} Å | "
          f"Range: {stats['min']:.3f}-{stats['max']:.3f} Å")

bulk_nn = 4.1932 * np.sqrt(3)/2  
print(f"\nBulk Na reference: {bulk_nn:.3f} Å")
