from gpaw import GPAW, PW, FermiDirac
from gpaw.poisson import PoissonSolver
from ase.build import bulk, surface
from ase.optimize import BFGS
import numpy as np
import matplotlib.pyplot as plt


CONVERGENCE_THRESHOLD = 0.01  # J/m²

a = 4.1932
Na_bulk = bulk('Na', 'bcc', a=a)

calc_bulk = GPAW(
    mode=PW(300), # change to 500 eV
    xc='PBE',
    kpts=(4, 4, 4), # change to (8, 8, 8)
    occupations=FermiDirac(0.01),
    txt='bulk_Na.txt'
)
Na_bulk.calc = calc_bulk
E_bulk = Na_bulk.get_potential_energy() / len(Na_bulk)

facets = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
layers_list = [3,4] #layers_list = [10, 12, 14, 16, 18]
vacuum_list = [5, 8]#vacuum_list = [20, 25, 30, 35]

all_gammas = {facet: [] for facet in facets}
all_params = {facet: [] for facet in facets}

for facet in facets:
    best_gamma = None
    best_diff = float('inf')
    for layers in layers_list:
        for vacuum in vacuum_list:
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
                print(f"Error relaxing slab for {facet} 
                      with {layers} layers and {vacuum} Å vacuum: {e}")
                continue

            E_slab = slab.get_potential_energy()
            a1, a2 = slab.cell[:2]
            A = np.linalg.norm(np.cross(a1, a2))

            gamma = (E_slab - len(slab) * E_bulk) * 1.60218e-19 / (2 * A * 1e-20)

            all_gammas[facet].append(gamma)
            all_params[facet].append((layers, vacuum))

            if best_gamma is not None:
                diff = abs(gamma - best_gamma)
                if diff < best_diff:
                    best_diff = diff
                    best_gamma = gamma
                if diff < CONVERGENCE_THRESHOLD:
                    print(f"Converged for {facet} at layers={layers}, 
                          vacuum={vacuum} with diff={diff:.4f} J/m²")
            else:
                best_gamma = gamma

    print(f"{facet}: {best_gamma:.3f} J/m²")

# plot surface energy vs layers for each facet
for facet in facets:
    for vacuum in vacuum_list:
        filtered_indices = [i for i, (layers, vac) in enumerate(all_params[facet]) 
                            if vac == vacuum]
        filtered_layers = [layers for i, (layers, vac) in enumerate(all_params[facet]) 
                           if vac == vacuum]
        filtered_gammas = [all_gammas[facet][i] for i in filtered_indices]

        plt.figure()
        plt.plot(filtered_layers, filtered_gammas, marker='o')
        if len(filtered_gammas) >= 2:
            plt.scatter(filtered_layers[-2:], filtered_gammas[-2:], color='red', marker='*')
            for i in range(-2, 0):
                plt.text(filtered_layers[i], filtered_gammas[i], f'{filtered_gammas[i]:.3f}', 
                         ha='left', va='bottom')
        plt.title(f'Surface Energy vs Layers for {facet} Facet (Vacuum = {vacuum} Å)')
        plt.xlabel('Layers')
        plt.ylabel('Surface Energy (J/m²)')
        plt.grid(True)
        plt.savefig(f'surface_energy_vs_layers_{facet}_vacuum_{vacuum}.png')
        plt.close()

# plot surface energy vs vacuum for each facet
for facet in facets:
    for layers in layers_list:
        filtered_indices = [i for i, (lay, vacuum) in enumerate(all_params[facet]) 
                            if lay == layers]
        filtered_vacuum = [vacuum for i, (lay, vacuum) in enumerate(all_params[facet]) 
                           if lay == layers]
        filtered_gammas = [all_gammas[facet][i] for i in filtered_indices]

        plt.figure()
        plt.plot(filtered_vacuum, filtered_gammas, marker='o')
        if len(filtered_gammas) >= 2:
            plt.scatter(filtered_vacuum[-2:], filtered_gammas[-2:], color='red', marker='*')
            for i in range(-2, 0):
                plt.text(filtered_vacuum[i], filtered_gammas[i], f'{filtered_gammas[i]:.3f}', 
                         ha='left', va='bottom')
        plt.title(f'Surface Energy vs Vacuum for {facet} Facet (Layers = {layers})')
        plt.xlabel('Vacuum (Å)')
        plt.ylabel('Surface Energy (J/m²)')
        plt.grid(True)
        plt.savefig(f'surface_energy_vs_vacuum_{facet}_layers_{layers}.png')
        plt.close()