from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
from ase.io import read, write
from gpaw import GPAW, PW
import os
import numpy as np
import matplotlib.pyplot as plt
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "results") 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. EMT 
# New H20 molecule
atoms_emt = molecule('H2O')
atoms_emt.set_cell((6.0, 6.0, 6.0))
atoms_emt.center()

calc_emt = EMT()
atoms_emt.calc = calc_emt
emt_traj_file = os.path.join(output_dir, '1_h2o_emt.traj')
opt_emt = QuasiNewton(atoms_emt, trajectory=emt_traj_file)
opt_emt.run(fmax=0.05)

atoms_emt = read(emt_traj_file)
o_h1_length_emt = atoms_emt.get_distance(0, 1)
o_h2_length_emt = atoms_emt.get_distance(0, 2)
h1_o_h2_angle_emt = atoms_emt.get_angle(1, 0, 2)

print('EMT:')
print(f'1. O-H1 bond length: {o_h1_length_emt:.5f}, O-H2 bond length: {o_h2_length_emt:.5f}, bond angle: {h1_o_h2_angle_emt:.5f} degrees')

# 2. GPAW
# New H2O molecule
atoms_gpaw = molecule('H2O')
atoms_gpaw.set_cell((10.0, 10.0, 10.0))
atoms_gpaw.center()

calc_gpaw = GPAW(mode='fd', txt=os.path.join(output_dir, '1_h2o_fd.txt'), h=0.2, xc='PBE')
atoms_gpaw.calc = calc_gpaw
gpaw_traj_file = os.path.join(output_dir, '1_h2o_gpaw.traj')
opt_gpaw = QuasiNewton(atoms_gpaw, trajectory=gpaw_traj_file)
opt_gpaw.run(fmax=0.05)

atoms_gpaw = read(gpaw_traj_file)
o_h1_length_gpaw = atoms_gpaw.get_distance(0, 1)
o_h2_length_gpaw = atoms_gpaw.get_distance(0, 2)
h1_o_h2_angle_gpaw = atoms_gpaw.get_angle(1, 0, 2)

print('GPAWs:')
print(f'1. O-H1 bond length: {o_h1_length_gpaw:.5f} , O-H2 bond length: {o_h2_length_gpaw:.5f}, H1-O-H2 bond angle: {h1_o_h2_angle_gpaw:.5f} degrees')

dist = atoms_gpaw.get_distance(0 ,1) # The attrbute.get_distance takes two arguments. The argumentsare the atom number in the system.

#print('The distance between atoms O and H is ' , f'{dist:.5f} ')
angle = atoms_gpaw.get_angle(1 , 0 , 2)
#print ( 'The angle between atoms O , the two H is' , f'{angle:.5f} ')


def calculate_atomization_energy(mode, xc, h=None, pw=None , a = 8.0):
    """
    Calculate the atomization energy of H2O for a given basis set mode, XC functional, and grid-spacing.
    :param mode: Basis set mode ('fd', 'PW(400)', or 'lcao')
    :param xc: Exchange-correlation functional (e.g., 'LDA', 'PBE', 'RPBE')
    :param h: Grid-spacing (only for fd mode)
    :param a: Lattice constant 
    :return: Atomization energy of H2O
    """
    energies = {}
    start_time = time.time()
    for name in ['H2O', 'H', 'O']:
        system = molecule(name)
        system.set_cell((a, a, a))
        system.center()
        
        # GPAW calculator setup based on mode
        if mode == 'fd':  # Finite Difference
            if h is None:
                raise ValueError("For 'fd' mode, h must be specified.")
            calc = GPAW(mode='fd',  h=h, xc=xc, txt=os.path.join(output_dir, f"1_{name}_{mode}_{xc}_{h}.txt"))
        elif mode == 'PW':  # Plane-Wave with 400 eV cutoff
            if pw is None:
                raise ValueError("For 'PW' mode, pw (cutoff energy) must be specified.")
            calc = GPAW(mode=PW(pw),  xc=xc, txt=os.path.join(output_dir, f"1_{name}_{mode}_{xc}_{pw}.txt"))
        elif mode == 'lcao':  # LCAO
            calc = GPAW(mode='lcao', basis = 'dzp', xc=xc, txt=os.path.join(output_dir, f"1_{name}_{mode}_{xc}.txt"))
        else:
            raise ValueError("Invalid mode! Choose from 'fd', 'PW', or 'lcao'.")

        system.calc = calc
        energy = system.get_potential_energy()
        energies[name] = energy
    
    end_time = time.time() 
    elapsed_time = end_time - start_time 
    atomization_energy = energies['H2O'] - 2 * energies['H'] - energies['O']
    
    return atomization_energy, elapsed_time


# 2. Calulate the atomization energy of H2O using LDA, PBE, and RPBE 
a = 8.0

xc_functionals = ['LDA', 'PBE', 'RPBE']
for xc in xc_functionals:
    e_atomization, _ = calculate_atomization_energy('fd' , xc, h=0.2, pw = None,a=a) # use 'fd' as mode
    print(f"2.XC - functional: {xc}, Atomization energy of H2O: {e_atomization} eV")

#3. Calculate atomization energy of H2O when using the three different basis sets option
xc = 'PBE'  # use PBE functional
modes = ['fd', 'PW', 'lcao']
for mode in modes:
    e_atomization , _  = calculate_atomization_energy(mode, xc, h=0.2, pw=400, a=a)
    print(f"3. Basis Set: {mode}, Atomization Energy of H2O: {e_atomization:.5f} eV")


#4. Plot the convergence of the atomization energy with respect to the grid - spacing h.

# 4.1 Grid-spacing
fig, ax1 = plt.subplots(figsize=(8, 6)) 
grid_spacings = [0.16, 0.18, 0.20, 0.22, 0.24, 0.26]
xc = 'LDA'
mode = 'fd'
atomization_energies = []
computation_times = []

for h in grid_spacings:
    e_atomization , elapsed_time = calculate_atomization_energy(mode, xc, h, pw=None, a=a)
    atomization_energies.append(e_atomization)
    computation_times.append(elapsed_time) 
    print(f"Grid-spacing h: {h}, Atomization energy: {e_atomization:.5f} eV, Time: {elapsed_time:.2f} s")

x = np.array(grid_spacings)
y1 = np.array(atomization_energies)
y2 = np.array(computation_times)

print(f"DEBUG: x = {x}, shape = {x.shape}")
print(f"DEBUG: y2 = {y2}, shape = {y2.shape}")

ax1.plot(x, y1, 'b-o', label="Atomization Energy")
ax1.scatter(x, y1, color='b', marker='o', s=60)  
ax1.set_xlabel('Grid-spacing h [Å]')
ax1.set_ylabel('Atomization energy of H2O [eV]', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(x, y2, 'r-s', label="Computation Time")
ax2.scatter(x, y2, color='r', marker='s', s=60)
ax2.set_ylabel('Time [s]', color='r')
ax2.tick_params(axis='y', labelcolor='r')

lines_1, labels_1 = ax1.get_legend_handles_labels() 
lines_2, labels_2 = ax2.get_legend_handles_labels()  
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
plt.title('Convergence of Atomization Energy with Grid-spacing')
plt.grid(True)
plt.savefig(os.path.join(output_dir, '4_1_convergence_plot_h.png'))


#4.2 plane-wave

fig, ax1 = plt.subplots(figsize=(8, 6)) 
pw_cutoffs = [200, 250, 300, 350, 400, 450]

xc = 'LDA'
mode = 'PW'
atomization_energies = []
computation_times = []

for pw in pw_cutoffs:
    e_atomization , elapsed_time = calculate_atomization_energy(mode, xc, h=None, pw=pw, a=a)
    atomization_energies.append(e_atomization)
    computation_times.append(elapsed_time)
    print(f"PW cutoff energy: {pw}, Atomization energy: {e_atomization:.5f} eV, Time: {elapsed_time:.2f} s")


x = np.array(pw_cutoffs)
y1 = np.array(atomization_energies)
y2 = np.array(computation_times)

ax1.plot(x, y1, 'b-o', label="Atomization Energy")
ax1.scatter(x, y1, color='b', marker='o', s=60)  
ax1.set_xlabel('Plane-wave cutoff energy [eV]')    
ax1.set_ylabel('Atomization energy of H2O [eV]', color='b')
ax1.tick_params(axis='y', labelcolor='b')

#print(f"DEBUG: x = {x}, shape = {x.shape}")
#print(f"DEBUG: y2 = {y2}, shape = {y2.shape}")


ax2 = ax1.twinx()
ax2.plot(x, y2, 'r-s', label="Computation Time")
ax2.scatter(x, y2, color='r', marker='s', s=60)
ax2.set_ylabel('Time [s]', color='r')
ax2.tick_params(axis='y', labelcolor='r')

lines_1, labels_1 = ax1.get_legend_handles_labels() 
lines_2, labels_2 = ax2.get_legend_handles_labels()  
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
plt.title('Convergence of Atomization Energy with plane-wave')
plt.grid(True)
plt.savefig(os.path.join(output_dir, '4_2_convergence_plot_pw.png'))

