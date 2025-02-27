import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk
from ase.eos import EquationOfState
from gpaw import GPAW, PW

# Different lattice parameters
a_values = np.linspace(4.0, 4.5, 9)  #  10 values between 4.0 and 4.5 Å
energies = []

for a in a_values:
    atoms = bulk('Na', 'bcc', a=a) 
    calc = GPAW(
        mode=PW(500),        # from task 1
        xc='PBE',
        setups={'Na': '1'},  #  3s¹ 
        kpts=(8, 8, 1),      # from task 1
        txt=f'./A3/task2_Na_a_{a:.2f}.out'
    )
    
    atoms.set_calculator(calc)
    energy = atoms.get_potential_energy()
    energies.append(energy)
    print(f'Lattice parameter a = {a:.2f} Å, Energy = {energy:.6f} eV')

# Volume calculation for bcc structure
volumes = a_values**3 / 2  # two atoms in bcc unit cell (V = a³/2)

# use ASE's EquationOfState class to fit the data
eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()  # v0: equilibrium volume, e0: equilibrium energy, B: bulk modulus (eV/Å³)

# latice parameter at equilibrium
a_eq = (2 * v0) ** (1/3)  #  V = a³/2 
B_GPa = B * 160.217  # transfer eV/Å³ to GPa

print(f"\nFinal Results:")
print(f"Equilibrium Lattice Parameter (a) = {a_eq:.4f} Å")
print(f"Bulk Modulus (B) = {B_GPa:.4f} GPa")

# plot the EOS
plt.figure(figsize=(10, 6))
eos.plot()  

for v, e in zip(volumes, energies):
    plt.scatter(v, e, color='red') 
    plt.annotate(f"({v:.4f}, {e:.4f})", (v, e), 
                 textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)

plt.xlabel('Volume (Å³)')
plt.ylabel('Energy (eV)')
plt.title('Fitting Na EOS')
plt.grid(True)
plt.savefig('./A3/task2_Na_EOS.png', dpi=300)


# Plot lattice constant vs energy
plt.figure(figsize=(10, 6))
plt.plot(a_values, energies, marker='o', color='blue')
plt.xlabel('Lattice Constant (Å)')
plt.ylabel('Energy (eV)')
plt.title('Lattice Constant vs Energy for BCC Na')
plt.grid(True)

# Mark the equilibrium point
plt.scatter(a_eq, e0, color='red', label=f'Equilibrium a = {a_eq:.4f} Å')
plt.legend()
plt.savefig('./A3/task2_Na_Lattice_vs_Energy.png', dpi=300)
plt.show()
