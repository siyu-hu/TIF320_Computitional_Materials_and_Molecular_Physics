from gpaw import GPAW, PW, FermiDirac
from gpaw.poisson import PoissonSolver
from ase.build import bulk, surface
from ase.optimize import BFGS
import numpy as np
import os
from mpi4py import MPI
import pandas as pd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


script_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(script_dir, 'result')
if rank == 0:
    os.makedirs(result_dir, exist_ok=True)


CONVERGENCE_THRESHOLD = 0.01  # J/m²

a = 4.1932
Na_bulk = bulk('Na', 'bcc', a=a)
calc_bulk = GPAW(
        mode=PW(500),
        xc='PBE',
        kpts=(8, 8, 8),
        occupations=FermiDirac(0.01),
        txt=os.path.join(result_dir, 'bulk_Na.txt')
    )

Na_bulk.calc = calc_bulk
# calculate bulk energy

E_bulk = Na_bulk.get_potential_energy() / len(Na_bulk)

facets = [(1, 0, 0), (1, 1, 0)] #facets = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
layers_list = [3] #layers_list = [10, 12, 14, 16, 18]
vacuum_list = [5, 8] #vacuum_list = [15, 20, 25, 30, 35]


tasks = [(facet, layers, vacuum) for facet in facets for layers in layers_list for vacuum in vacuum_list]
# Scatter tasks to all ranks
my_tasks = tasks[rank::size]
print(f"Rank {rank} has {len(my_tasks)} tasks")

local_results = []
for facet, layers, vacuum in my_tasks:
    print(f"Rank {rank}: Calculating for {facet} with layers={layers}, vacuum={vacuum} Å")

    slab = surface(Na_bulk, facet, layers=layers)
    slab.center(vacuum=vacuum, axis=2)
    slab.pbc = (True, True, False)

    calc_s = GPAW(
        mode=PW(500),
        xc='PBE',
        kpts=(8, 8, 1),
        occupations=FermiDirac(0.01),
        txt=os.path.join(result_dir, f"slab_{facet}_{layers}_{vacuum}.txt")
    )
    slab.calc = calc_s

    relax = BFGS(slab)
    try:
        relax.run(fmax=0.01)
    except Exception as e:
        print(f"Rank {rank}: Error relaxing slab {facet} with {layers} layers, {vacuum} Å vacuum: {e}")
        
        continue

    # calculate surface energy
    E_slab = slab.get_potential_energy()
    a1, a2 = slab.cell[:2]
    A = np.linalg.norm(np.cross(a1, a2))

    gamma = (E_slab - len(slab) * E_bulk) * 1.60218e-19 / (2 * A * 1e-20)

    local_results.append({
        'facet_name': str(facet),
        'vacuum': vacuum,
        'layers': layers,
        'surface_energy': gamma,
    })

all_results = comm.gather(local_results, root=0)

if rank == 0:
    all_results_flat = [item for sublist in all_results for item in sublist]
    df = pd.DataFrame(all_results_flat)
    print(df)
    df.to_csv(os.path.join(result_dir, 'surface_energy_results.csv'), index=False)