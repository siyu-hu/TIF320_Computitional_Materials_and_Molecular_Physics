import matplotlib.pyplot as plt
import numpy as np

def read_wavefunction(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)  
    r = data[:, 0]  
    psi = data[:, 1] 
    return r, psi

r_task4, psi_task4 = read_wavefunction("./A5/task4_helium_wavefunction_and_energy.csv")
r_task5, psi_task5 = read_wavefunction("./A5/task5_helium_wavefunction_and_energy.csv")
r_task6, psi_task6 = read_wavefunction("./A5/task6_helium_wavefunction_and_energy.csv")
r_task7, psi_task7 = read_wavefunction("./A5/task7_helium_wavefunction_and_energy.csv")


plt.figure(figsize=(10, 6))
plt.plot(r_task4, psi_task4, label='Task 4: No exchange, no correlation', color='blue')
plt.plot(r_task5, psi_task5, label='Task 5: Exchange, no correlation', color='red')
plt.plot(r_task6, psi_task6, label='Task 6: No exchange, correlation', color='green')
plt.plot(r_task7, psi_task7, label='Task 7: Exchange and correlation', color='purple')

plt.xlabel('Radial distance r (a.u.)')
plt.ylabel('Wavefunction')
plt.title('Helium Atom Ground State Wavefunction for Different Tasks')
plt.legend()
plt.grid(True)
plt.savefig(f'./A5/task8_helium_wavefunction_comparison.png')
