from gpaw import GPAW, PW, FermiDirac
from gpaw.poisson import PoissonSolver
from ase.build import bulk, surface
from ase.optimize import BFGS
import numpy as np
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(script_dir,'result')
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

facets = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
layers_list = [10, 12, 14, 16, 18, 20]
vacuum_list = [15, 20, 25, 30, 35, 40]

tasks = [(facet, layers, vacuum) for facet in facets for layers in layers_list for vacuum in vacuum_list]

local_results = []
for facet, layers, vacuum in tasks:
    print(f"Calculating for {facet} with layers={layers}, vacuum={vacuum} Å")

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
        print(f"Error relaxing slab {facet} with {layers} layers, {vacuum} Å vacuum: {e}")
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

df = pd.DataFrame(local_results)
print(df)
df.to_csv(os.path.join(result_dir,'surface_energy_results.csv'), index=False)