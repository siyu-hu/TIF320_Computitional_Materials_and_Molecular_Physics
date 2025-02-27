import os
import numpy as np
import csv
from gpaw import GPAW, PW, FermiDirac
from gpaw.poisson import PoissonSolver
from ase.build import bulk, surface
from ase.optimize import BFGS
from mpi4py import MPI

def calculate_bulk_energy(results_folder):
    a = 4.1932
    Na_bulk = bulk('Na', 'bcc', a=a)
    bulk_txt_path = os.path.join(results_folder, 'bulk_Na.txt')
    calc_bulk = GPAW(
        mode=PW(300), #mode=PW(500),
        xc='PBE',
        kpts=(4, 4, 4),#kpts=(8, 8, 8),
        occupations=FermiDirac(0.01),
        txt=bulk_txt_path,
        parallel={'domain': 1}
    )
    Na_bulk.calc = calc_bulk
    return Na_bulk.get_potential_energy() / len(Na_bulk)

def create_and_relax_slab(Na_bulk, plane, layers, vacuum,results_folder):
    slab = surface(Na_bulk, plane, layers=layers)
    slab.center(vacuum=vacuum, axis=2)
    slab.pbc = (True, True, False)
    slab_txt_path = os.path.join(results_folder, f"slab_{plane}_{layers}_{vacuum}.txt")
    calc_s = GPAW(
        mode=PW(300), #mode=PW(500),
        xc='PBE',
        kpts=(4, 4, 1), #kpts=(8, 8, 1),
        occupations=FermiDirac(0.01),
        txt=slab_txt_path,
        parallel={'domain': 1} 
    )
    slab.calc = calc_s
    relax = BFGS(slab)
    relax.run(fmax=0.01)
    return slab

def calculate_surface_energy(slab, E_bulk):
    E_slab = slab.get_potential_energy()
    a1, a2 = slab.cell[:2]
    A = np.linalg.norm(np.cross(a1, a2))
    return (E_slab - len(slab) * E_bulk) * 1.60218e-19 / (2 * A * 1e-20)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    results_folder = os.path.join(os.getcwd(), 'task6_new_mpi_results')
    os.makedirs(results_folder, exist_ok=True)

    E_bulk = calculate_bulk_energy(results_folder)
    vacuum_list = [3]  # vacuum_list = [15, 20, 25, 30, 35]
    layers_list = [3, 5]  # layers_list = [10, 12, 14, 16, 18]

    total_combinations = len(vacuum_list) * len(layers_list)
    combinations_per_process = total_combinations // size

    start_idx = rank * combinations_per_process
    end_idx = start_idx + combinations_per_process if rank != size - 1 else total_combinations
    csv_file = os.path.join(results_folder, 'surface_energy_results.csv')

    a = 4.1932
    Na_bulk = bulk('Na', 'bcc', a=a)

    surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]

    process_results = []
    #print(f"Process {rank}: Handling combinations from {start_idx} to {end_idx}", flush=True)

    for idx in range(start_idx, end_idx):
        vacuum = vacuum_list[idx // len(layers_list)]
        layers = layers_list[idx % len(layers_list)]

        surface_energies = []
        print(f"Process {rank}: START: Vacuum: {vacuum:.1f} Å, Layers: {layers}", flush=True)
        for plane in surfaces:
            slab = create_and_relax_slab(Na_bulk, plane, layers, vacuum, results_folder)
            surface_energy = calculate_surface_energy(slab, E_bulk)
            surface_energies.append(surface_energy)

        process_results.append([vacuum, layers] + surface_energies)
        print(f"Process {rank}: END: Vacuum: {vacuum} Å, Layers: {layers},"
            f"(100): {surface_energies[0]:.6f} J/m², "
            f"(110): {surface_energies[1]:.6f} J/m², "
            f"(111): {surface_energies[2]:.6f} J/m²", flush=True)

    all_results = comm.gather(process_results, root=0)

    if rank == 0:
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Vacuum Layer (Å)', 'Layers', '(100) Surface Energy (J/m²)', '(110) Surface Energy (J/m²)',
                             '(111) Surface Energy (J/m²)'])
            for result in all_results:
                for row in result:
                    writer.writerow(row)



if __name__ == "__main__":
    main()
