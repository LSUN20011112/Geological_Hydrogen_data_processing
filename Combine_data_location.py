import numpy as np
from pathlib import Path

# === Define tools function ===

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

def read_mesh_header(mesh_file_path):
    with open(mesh_file_path) as f:
        lines = f.readlines()
    nx, ny, nz = map(int, lines[0].split())
    x0, y0, z0 = map(float, lines[1].split())
    hx = parse_cell_sizes(lines[2])
    hy = parse_cell_sizes(lines[3])
    hz = parse_cell_sizes(lines[4])
    return (nx, ny, nz), (x0, y0, z0), (hx, hy, hz)

def compute_cell_centers(h_sizes, origin):
    centers = []
    pos = origin
    for h in h_sizes:
        centers.append(pos + h / 2)
        pos += h
    return np.array(centers)

# === Main Processing Functions ===

def generate_xyz_from_models(diff_path, dens_path, susc_path, mesh_path, output_xyz_path):
    # load model
    diff = np.load(diff_path)
    dens = np.load(dens_path)
    susc = np.load(susc_path)

    # check shapes
    if not (diff.shape == dens.shape == susc.shape):
        raise ValueError("The models have inconsistent shapes and cannot be merged.")

    nx, ny, nz = diff.shape
    (mx, my, mz), (x0, y0, z0), (hx, hy, hz) = read_mesh_header(mesh_path)

    if (nx, ny, nz) != (mx, my, mz):
        raise ValueError("The model size does not match the mesh size.")

    # compute the cell centers
    x_centers = compute_cell_centers(hx, x0)
    y_centers = compute_cell_centers(hy, y0)
    z_centers = compute_cell_centers(hz, z0)

    X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")

    # flatten and merge
    points = np.column_stack([
        X.flatten(order="C"),
        Y.flatten(order="C"),
        Z.flatten(order="C"),
        diff.flatten(order="C"),
        dens.flatten(order="C"),
        susc.flatten(order="C")
    ])

    # save as .xyz and .npy
    np.savetxt(output_xyz_path, points, fmt="%.4f", header="X Y Z Diff Density Susc")
    np.save(output_npy_path, points)
    print(f"Export successfully：{output_xyz_path} and {output_npy_path}")

# === paths ===

diff_path  = r"C:\Users\sunlo\Desktop\Geological_Hydrogen\diff_model.npy"
dens_path  = r"C:\Users\sunlo\Desktop\Geological_Hydrogen\joint_dens_model.npy"
susc_path  = r"C:\Users\sunlo\Desktop\Geological_Hydrogen\joint_susc_model.npy"
mesh_path  = r"C:\Users\sunlo\Desktop\Geological_Hydrogen\mesh_joint_Hannah_exfine_rm.txt"
output_xyz_path = r"C:\Users\sunlo\Desktop\Geological_Hydrogen\merged_model_output.xyz"
output_npy_path = r"C:\Users\sunlo\Desktop\Geological_Hydrogen\merged_model_output.npy"

# === run ===
generate_xyz_from_models(diff_path, dens_path, susc_path, mesh_path, output_xyz_path)




