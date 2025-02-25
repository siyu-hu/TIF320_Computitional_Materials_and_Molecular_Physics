from ase.build import bulk
from ase.build import surface
from gpaw import GPAW, PW
import numpy as np
import matplotlib.pyplot as plt

na_bulk = bulk('Na', 'bcc', a=4.1329)

calc = GPAW(mode=PW(300), # change to 600 eV
            kpts=(3, 3, 3), # change to (6, 6, 6)
            txt='task6_na_bulk.txt')

na_bulk.calc = calc

try:
    E_bulk_total = na_bulk.get_potential_energy()
    E_bulk = E_bulk_total / len(na_bulk)
except Exception as e:
    print(f"Error calculating bulk energy: {e}")
    raise

facets = [(1, 0, 0), (1, 1, 1), (1, 1, 0)]
layers_list = [3, 4]
vacuum_list = [5, 8]

gamma_all_results = {facet: [] for facet in facets}
params_all_results = {facet: [] for facet in facets}

gamma_results = {}

for facet in facets:
    best_gamma = None
    best_diff = float('inf')
    for layers in layers_list:
        for vacuum in vacuum_list:
            slab = surface(na_bulk, facet, layers=layers, vacuum=vacuum)
            slab.calc = GPAW(mode=PW(300),
                             kpts=(3, 3, 1),
                             txt=f'na_{facet}_{layers}_{vacuum}.txt')
            try:
                E_slab = slab.get_potential_energy()
            except Exception as e:
                print(f"Error calculating surface energy for {facet} with {layers} layers and {vacuum} Å vacuum: {e}")
                continue

            N = len(slab)
            cell = slab.get_cell()

            if facet == (1, 0, 0):
                A = cell[0, 0] * cell[1, 1]
            elif facet == (1, 1, 1):
                area_vec = np.cross(cell[0], cell[1])
                A = np.linalg.norm(area_vec)
            elif facet == (1, 1, 0):
                A = cell[0, 0] * cell[1, 1]

            # surface energy - gamma
            gamma = (E_slab - N * E_bulk) / (2 * A)

            gamma_all_results[facet].append(gamma)
            params_all_results[facet].append((layers, vacuum))

            if best_gamma is not None:
                diff = abs(gamma - best_gamma)
                if diff < best_diff:
                    best_diff = diff
                    best_gamma = gamma
            else:
                best_gamma = gamma

    gamma_results[facet] = best_gamma

for facet, gamma in gamma_results.items():
    print(f"Surface energy of {facet} facet: {gamma} eV/Å²")

with open('surface_energy_results_local.txt', 'w') as f:
    f.write("Facet\tSurface Energy (eV/Å²)\n")
    for facet, gamma in gamma_results.items():
        f.write(f"{facet}\t{gamma}\n")

# plot surface energy vs layers for each facet
for facet in facets:
    for vacuum in vacuum_list:
        filtered_indices = [i for i, (layers, vac) in enumerate(params_all_results[facet]) if vac == vacuum]
        filtered_layers = [layers_list[layers - min(layers_list)] for i, (layers, vac) in enumerate(params_all_results[facet]) if vac == vacuum]
        filtered_gammas = [gamma_all_results[facet][i] for i in filtered_indices]

        plt.figure()
        plt.plot(filtered_layers, filtered_gammas, marker='o')
        if len(filtered_gammas) >= 2:
            plt.scatter(filtered_layers[-2:], filtered_gammas[-2:], color='red', marker='*')
        plt.title(f'Surface Energy vs Layers for {facet} Facet (Vacuum = {vacuum} Å)')
        plt.xlabel('Layers')
        plt.ylabel('Surface Energy (eV/Å²)')
        plt.grid(True)
        plt.savefig(f'surface_energy_vs_layers_{facet}_vacuum_{vacuum}.png')
        plt.close()

# plot surface energy vs vacuum for each facet
for facet in facets:
    for layers in layers_list:
        filtered_indices = [i for i, (lay, vacuum) in enumerate(params_all_results[facet]) if lay == layers]
        filtered_vacuum = [vacuum_list[vac - min(vacuum_list)] for i, (lay, vacuum) in enumerate(params_all_results[facet]) if lay == layers]
        filtered_gammas = [gamma_all_results[facet][i] for i in filtered_indices]

        plt.figure()
        plt.plot(filtered_vacuum, filtered_gammas, marker='o')
        if len(filtered_gammas) >= 2:
            plt.scatter(filtered_vacuum[-2:], filtered_gammas[-2:], color='red', marker='*')
        plt.title(f'Surface Energy vs Vacuum for {facet} Facet (Layers = {layers})')
        plt.xlabel('Vacuum (Å)')
        plt.ylabel('Surface Energy (eV/Å²)')
        plt.grid(True)
        plt.savefig(f'surface_energy_vs_vacuum_{facet}_layers_{layers}.png')
        plt.close()