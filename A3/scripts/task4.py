from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from ase import Atoms as ASEAtoms
from ase.build import bulk
from gpaw import GPAW, PW
import time
import numpy as np
import matplotlib.pyplot as plt

print("\n=== Starting Phonon Calculation for BCC Na ===")
start_time = time.time()

# initialize BCC Na unit cell
na = bulk('Na', 'bcc', a=4.1932)
print("[1/6] Original BCC unit cell created (2 atoms).")

calc = GPAW(
    mode=PW(300),
    xc='PBE',
    kpts=(4, 4, 4),
    random=True,
    setups={'Na': '1'},
    symmetry='off',  
    txt='./A3/task4_Na_phonopy.txt'
)
print("[2/6] GPAW calculator configured (kpts=4x4x4, cutoff=600 eV).")

# initialize phonopy object
phonon = Phonopy(
    PhonopyAtoms(
        symbols=na.symbols,
        positions=na.positions,
        cell=na.cell
    ),
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    is_symmetry=False
)
print("[3/6] 2x2x2 supercell created (16 atoms).")

# Generate displacement configurations
phonon.generate_displacements(distance=0.01)
supercells = phonon.supercells_with_displacements
n_displacements = len(supercells)
print(f"[4/6] Generated {n_displacements} displacement configurations.")

# calculate forces
print(f"\n--- Calculating Forces ({n_displacements} configurations) ---")
force_list = []
for i, scell_phonopy in enumerate(supercells, 1):
    scell_ase = ASEAtoms(
        symbols=scell_phonopy.symbols,
        positions=scell_phonopy.positions,
        cell=scell_phonopy.cell,
        pbc=True
    )
    scell_ase.calc = calc
    print(f"  Processing displacement {i}/{n_displacements}...")
    forces = scell_ase.get_forces()
    force_list.append(forces)

phonon.forces = force_list
print("\n[5/6] All forces calculated. Building force constants...")
phonon.produce_force_constants()

# define the path for band structure calculation
path = [
    ("G", [0.0, 0.0, 0.0]),
    ("H", [0.0, 0.0, 1.0]),
    ("P", [0.5, 0.5, 0.5]),
    ("N", [0.5, 0.5, 0.0]),
    ("G", [0.0, 0.0, 0.0])
]

print("[6/6] Computing phonon band structure along G-H-P-N-G...")
phonon.run_band_structure(
    paths=[[path[0][1], path[1][1]],
          [path[1][1], path[2][1]],
          [path[2][1], path[3][1]],
          [path[3][1], path[4][1]]],
    labels=[point[0] for point in path]
)

#  Check if the phonon spectrum is stable
# frequencies = phonon.get_band_structure_dict()['frequencies']
# if np.any(frequencies < -1e-4): 
#     print("Warning: Imaginary frequencies detected!")
# else:
#     print("Phonon spectrum is dynamically stable.")


band_dict = phonon.get_band_structure_dict()
frequencies = band_dict['frequencies']  
qpoints = band_dict['qpoints'] 
labels = band_dict['labels']  


fig, ax = plt.subplots(figsize=(8, 6))

# 绘制声子谱
for i in range(frequencies.shape[1]):  # 遍历所有声子支
    ax.plot(qpoints, frequencies[:, i], color='blue', linewidth=1.5, label="Acoustic Branches" if i == 0 else "")

# 添加标题
ax.set_title("Phonon Band Structure of BCC Sodium", fontsize=14, pad=20)

# 添加坐标轴标签
ax.set_xlabel("Wave Vector", fontsize=12)
ax.set_ylabel("Frequency (THz)", fontsize=12)  # 明确频率单位

# 添加高对称点标签
ax.set_xticks([qpoints[i] for i in range(0, len(qpoints), len(qpoints)//4)])  # 根据路径段数设置
ax.set_xticklabels(labels, rotation=45, fontsize=10)

# 添加图例
ax.legend(loc="upper right", fontsize=10)

# 添加网格线和零频参考线
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax.grid(axis='y', linestyle=':', alpha=0.5, color='lightgray')

# 保存图像
fig.savefig(
    './A3/task4_Na_phonon_bandstructure_simple.png',
    dpi=300,  # 高分辨率
    bbox_inches='tight'  # 防止边缘被裁剪
)

# 完成
total_time = time.time() - start_time
print(f"\n=== Calculation Completed ===")
print(f"Total time: {total_time//60:.0f} min {total_time%60:.1f} sec")
print(f"Results saved to: ./A3/task4_Na_phonon_bandstructure_simple.png")