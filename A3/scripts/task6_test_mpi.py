from gpaw import GPAW, PW, FermiDirac
from gpaw.poisson import PoissonSolver
from ase.build import bulk, surface
from ase.optimize import BFGS
import numpy as np
import matplotlib.pyplot as plt
import os
from mpi4py import MPI
import pandas as pd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


script_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(script_dir, 'result')
os.makedirs(result_dir, exist_ok=True)


CONVERGENCE_THRESHOLD = 0.01  # J/m²


a = 4.1932
Na_bulk = bulk('Na', 'bcc', a=a)

calc_bulk = GPAW(
    mode=PW(300), # change to 500 eV
    xc='PBE',
    kpts=(4, 4, 4), # change to (8, 8, 8)
    occupations=FermiDirac(0.01),
    txt=os.path.join(result_dir, 'bulk_Na.txt')
)
Na_bulk.calc = calc_bulk
E_bulk = Na_bulk.get_potential_energy() / len(Na_bulk)

facets = [(1, 0, 0), (1, 1, 0)] #facets = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
layers_list = [3] #layers_list = [10, 12, 14, 16, 18]
vacuum_list = [5, 8] #vacuum_list = [20, 25, 30, 35]

all_gammas = {facet: [] for facet in facets}
all_params = {facet: [] for facet in facets}
results = []

for facet in facets:
    for layers in layers_list:
        for vacuum in vacuum_list:
            if rank == 0:
                print(f"BEGIN: Calculating for {facet} with layers= {layers}, vacuum= {vacuum} Å")
            slab = surface(Na_bulk, facet, layers=layers)
            slab.center(vacuum=vacuum, axis=2)
            slab.pbc = (True, True, False)

            calc_s = GPAW(
                mode=PW(500),
                xc='PBE',
                kpts=(8, 8, 1),
                occupations=FermiDirac(0.01),
                txt=None
            )
            slab.calc = calc_s

            relax = BFGS(slab)
            try:
                relax.run(fmax=0.01)
            except Exception as e:
                if rank == 0:
                    print(f"Error relaxing slab for {facet} with {layers} layers and {vacuum} Å vacuum: {e}")
                continue

            E_slab = slab.get_potential_energy()
            a1, a2 = slab.cell[:2]
            A = np.linalg.norm(np.cross(a1, a2))

            gamma = (E_slab - len(slab) * E_bulk) * 1.60218e-19 / (2 * A * 1e-20)

            all_gammas[facet].append(gamma)
            all_params[facet].append((layers, vacuum))

            results.append({
                'facet_name': str(facet),
                'vacuum': vacuum,
                'layers': layers,
                'surface_energy': gamma,
            })
        if rank == 0:
            print(f"FINISH: Calculating for {facet} with layers= {layers}, vacuum= {vacuum} Å")

df = pd.DataFrame(results)
print(df)
df.to_csv(os.path.join(result_dir, 'surface_energy_results.csv'), index=False)