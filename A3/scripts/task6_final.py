from gpaw import GPAW, FermiDirac
from ase.build import bcc100, bcc110, bcc111, bulk
from gpaw import GPAW, PW

def slab_energy(miller, N, k, a=4.1932, vacuum=10.0):

    if miller == (1, 0, 0):
        slab = bcc100('Na', (1, 1, N), a=a, vacuum=vacuum)
    elif miller == (1, 1, 0):
        slab = bcc110('Na', (1, 1, N), a=a, vacuum=vacuum)
    elif miller == (1, 1, 1):
        slab = bcc111('Na', (1, 1, N), a=a, vacuum=vacuum)
    else:
        raise ValueError("Only (100), (110), and (111) surfaces are supported.")

    slab.center(axis=2)  

    calc = GPAW(mode = PW(100), xc = 'PBE', kpts = (k, k, 1), setups ={ 'Na': '1'}, txt=None)
    slab.calc = calc
    e_slab = slab.get_potential_energy()
    calc.write(f'slab-{miller}.gpw')

    return e_slab, slab.cell.areas()[2], len(slab)

def bulk_energy(a=4.1932, k=8):
    Na_bulk = bulk('Na', 'bcc', a=a, cubic=True) 
    calc = GPAW(mode=PW(500),
                xc='PBE',
                kpts=(k, k, k),
                occupations=FermiDirac(0.01),
                setups ={ 'Na': '1'},
                txt=None)
    Na_bulk.calc = calc
    e_bulk = Na_bulk.get_potential_energy()
    calc.write('bulk.gpw')
    return e_bulk/Na_bulk.get_number_of_atoms()

e_bulk = bulk_energy()

N = 5
K = 14
surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]

for miller in surfaces:
    e_slab, A, num_atoms = slab_energy(miller, N, K)
    gamma = (e_slab - num_atoms * e_bulk) / (2 * A)* 16.0218
    print(f"Surface energy for Na {miller}: {gamma:.4f} J/m²")