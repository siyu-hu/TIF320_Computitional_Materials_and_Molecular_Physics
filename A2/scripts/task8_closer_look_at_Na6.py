import copy
from ase.io import read, write
from gpaw import GPAW, PW
from ase.optimize import BFGS

# Read standard structures
struct1 = read('standard_christmas_tree.xyz')
struct2 = read('standard_half_decahedron.xyz')

# Define at least 6 different parameter sets
parameter_sets = [
    {"mode": PW(350), "xc": "PBE"},          # Standard PW
    #{"mode": PW(400), "xc": "PBE"},   
    #{"mode": PW(500), "xc": "PBE"},        # Higher cutoff PW
    #{"mode": 'fd', "h": 0.2, "xc": "PBE"},   # Standard fd
    #{"mode": 'fd', "h": 0.15, "xc": "PBE"},  # Finer grid fd
    #{"mode": PW(350), "xc": "LDA"}  
    #{"mode": PW(400), "xc": "LDA"}
    #{"mode": 'lcao', "basis": "dzp", "xc": "PBE"},  # Double-zeta polarized
    # {"mode": 'lcao', "basis": "dzp", "xc": "LDA"}   
    #{"mode": 'lcao', "basis": "tzp", "xc": "PBE"},
    #{"mode": 'lcao', "basis": "qzp", "xc": "PBE"},
    #{"mode": PW(400), "xc": "HSE06"},
    #{"mode": PW(400), "xc": "PBE0"},
    #{"mode": PW(400), "xc": "SCAN"},

]

results = []

# Run calculations for both structures with all parameter sets
for struct_id, structure in enumerate([struct1, struct2], 1):
    for i, params in enumerate(parameter_sets):
        print(f"Running calculation for structure {struct_id} with params: {params}")
        calc_params = params.copy()
        mode = calc_params.pop("mode")

        calc = GPAW(
            mode=mode, 
            #h=params["h"],
            xc=params["xc"],
            #nbands=-10, 
            #kpts=(1, 1, 1),
            txt=f'task8_struct{struct_id}_calc_{i+1}.txt',
            #parallel={'domain': 1}
        )
        #structure.set_calculator(calc)
        structure.calc = calc
        opt = BFGS(structure)
        opt.run(fmax=0.01)
        energy = structure.get_potential_energy()
        results.append((struct_id, params, energy))
        write(f'task8_struct{struct_id}_relaxed_{i+1}.xyz', structure)

print("\nResults Table:")
print("{:<10} | {:<40} | {:<15}".format("Structure", "Parameters", "Energy (eV)"))
print("-" * 70)
for struct_id, params, energy in results:
    # Format parameters string based on mode type
    mode = params['mode']
    if isinstance(mode, PW):
        params_str = f"PW({mode.ecut}) eV, {params['xc']}"
    elif params['mode'] == 'fd':
        params_str = f"FD(h={params['h']}), {params['xc']}"
    elif params['mode'] == 'lcao':
        params_str = f"LCAO({params['basis']}), {params['xc']}"
    else:
        params_str = str(params)
        
    print("{:<10} | {:<40} | {:<15.6f}".format(f"Struct {struct_id}", params_str, energy))