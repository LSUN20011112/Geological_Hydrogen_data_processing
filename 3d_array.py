import numpy as np
from pathlib import Path

# find cell size
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

# read mesh file
def read_mesh_header(mesh_file_path):
    with open(mesh_file_path) as f:
        lines = f.readlines()

    nx, ny, nz = map(int, lines[0].split())
    x0, y0, z0 = map(float, lines[1].split())
    hx = parse_cell_sizes(lines[2])
    hy = parse_cell_sizes(lines[3])
    hz = parse_cell_sizes(lines[4])

    return (nx, ny, nz), (x0, y0, z0), (hx, hy, hz)

# main function: from 1D model + mesh, to 3D model and saved as .npy format
def convert_model_1d_to_3d(model_txt_path, mesh_txt_path, output_npy_path):
    # read
    model_1d = np.loadtxt(model_txt_path)

    # read mesh dimension
    (nx, ny, nz), _, _ = read_mesh_header(mesh_txt_path)

    # check dimension
    expected_size = nx * ny * nz
    if model_1d.size != expected_size:
        raise ValueError(f"model length {model_1d.size} different from mesh size({nx} × {ny} × {nz} = {expected_size})")

    # reshape → transpose rebuild (x, y, z) order
    model_3d = model_1d.reshape((nz, ny, nx), order='F')
    model_3d_xyz = np.transpose(model_3d, (2, 1, 0))  # shape: (nx, ny, nz)

    # save npy.
    np.save(output_npy_path, model_3d_xyz)
    print(f"Model saved as：{output_npy_path}")

    return model_3d_xyz


# === real work ===
if __name__ == "__main__":
    # replace the path one by one
    model_txt_path = Path(r"C:\Users\sunlo\Desktop\Geological_Hydrogen\My_work\GD_models\model_inpolygon.txt")
    mesh_txt_path = Path(r"C:\Users\sunlo\Desktop\Geological_Hydrogen/mesh_joint_Hannah_exfine_rm.txt")
    output_npy_path = Path(r"C:\Users\sunlo\Desktop\Geological_Hydrogen/diff_model.npy")

    model_3d = convert_model_1d_to_3d(model_txt_path, mesh_txt_path, output_npy_path)




