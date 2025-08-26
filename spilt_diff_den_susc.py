import numpy as np
from pathlib import Path

# === read head ===
def parse_cell_sizes(line):
    parts = line.strip().split()
    sizes = []
    for part in parts:
        if '*' in part:
            n, val = part.split('*')
            sizes.extend([float(val)] * int(n))
        else:
            sizes.append(float(part))
    return sizes

def read_mesh_header(mesh_path):
    with open(mesh_path) as f:
        lines = f.readlines()
    nx, ny, nz = map(int, lines[0].split())
    x0, y0, z0 = map(float, lines[1].split())
    hx = parse_cell_sizes(lines[2])
    hy = parse_cell_sizes(lines[3])
    hz = parse_cell_sizes(lines[4])
    return (nx, ny, nz), (x0, y0, z0), (hx, hy, hz)

# === split and write in UBC format ===
def split_and_save_ubc_txt(cleaned_npy_path, mesh_path, output_dir):
    # read data
    data = np.load(cleaned_npy_path)
    assert data.shape[1] == 6, "Expected (N, 6) array"

    (nx, ny, nz), *_ = read_mesh_header(mesh_path)
    N_expected = nx * ny * nz
    assert data.shape[0] == N_expected, f"Mismatch: {data.shape[0]} vs {N_expected}"

    # spilt 3 kinds of value
    diff_vals = data[:, 3]
    dens_vals = data[:, 4]
    susc_vals = data[:, 5]

    # reshape to 3D (nx, ny, nz)
    def reshape_to_3d(vals):
        return vals.reshape((nx, ny, nz), order="C")

    diff_3d = reshape_to_3d(diff_vals)
    dens_3d = reshape_to_3d(dens_vals)
    susc_3d = reshape_to_3d(susc_vals)

    # transpose to Fortran order 1D model
    def to_ubc_1d(model_3d):
        return np.transpose(model_3d, (2,1,0)).flatten(order='F')

    np.savetxt(Path(output_dir) / "diff_model_restored.txt", to_ubc_1d(diff_3d), fmt="%.6f")
    np.savetxt(Path(output_dir) / "density_model_restored.txt", to_ubc_1d(dens_3d), fmt="%.6f")
    np.savetxt(Path(output_dir) / "susc_model_restored.txt", to_ubc_1d(susc_3d), fmt="%.6f")

    print("Exported as UBC format .txt：")
    print(f"→ Diff:    {output_dir}/diff_model_restored.txt")
    print(f"→ Density: {output_dir}/density_model_restored.txt")
    print(f"→ Susc:    {output_dir}/susc_model_restored.txt")

# === Run run run! ===

split_and_save_ubc_txt(
    cleaned_npy_path=r"C:\Users\sunlo\Desktop\Geological_Hydrogen\My_work\Data_Cut\merged_model_cleaned.npy",
    mesh_path=r"C:\Users\sunlo\Desktop\Geological_Hydrogen\mesh_joint_Hannah_exfine_rm.txt",
    output_dir=r"C:\Users\sunlo\Desktop\Geological_Hydrogen\My_work\Data_Cut"
)






