import numpy as np
from discretize import TensorMesh

# ===== Paths =====
# UBC mesh path (.mesh)
mesh_path = r"C:\Users\sunlo\Desktop\Research\Modified_Geological_Hydrogen\mesh_joint_Hannah_exfine_rm.txt"

# UBC model path (.txt)
dens_path = r"C:\Users\sunlo\Desktop\Research\Modified_Geological_Hydrogen\07072025\den_susc_models\joint_dens_model_rm_UBC.txt"
susc_path = r"C:\Users\sunlo\Desktop\Research\Modified_Geological_Hydrogen\07072025\den_susc_models\joint_susc_model_rm_UBC.txt"
diff_path = r"C:\Users\sunlo\Desktop\Research\Modified_Geological_Hydrogen\07072025\quasi_model_output\model_inpolygon.txt"

# output path
output_npy = r"C:\Users\sunlo\Desktop\Research\Modified_Geological_Hydrogen\merged_model.npy"
output_txt = r"C:\Users\sunlo\Desktop\Research\Modified_Geological_Hydrogen\merged_model.xyz"

# ===== read mesh =====
mesh_rm = TensorMesh.read_UBC(mesh_path)
print("Mesh dimensions (nx, ny, nz):", mesh_rm.shape_cells)
print("Number of cells:", mesh_rm.nC)

# ===== read model =====
model_dens_rm_f = TensorMesh.read_model_UBC(mesh_rm, dens_path)
model_susc_rm_f = TensorMesh.read_model_UBC(mesh_rm, susc_path)
model_diff_rm_f = TensorMesh.read_model_UBC(mesh_rm, diff_path)

print("Model shapes:", model_dens_rm_f.shape, model_susc_rm_f.shape, model_diff_rm_f.shape)

# ===== stack to (N, 6) array =====
xyz = mesh_rm.cell_centers  # (N, 3)
merged_array = np.column_stack([xyz, model_diff_rm_f, model_dens_rm_f, model_susc_rm_f])

print("Merged array shape:", merged_array.shape)
print("First 5 rows:\n", merged_array[:5])

# ===== save array =====
np.save(output_npy, merged_array)
np.savetxt(output_txt, merged_array, fmt="%.6f")

print("Saved to:\n", output_npy, "\n", output_txt)
