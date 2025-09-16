import numpy as np
from pathlib import Path
from discretize import TensorMesh

def split_and_save_with_simpeg(cleaned_npy_path, mesh_path, output_dir):
    # === Read mesh ===
    mesh = TensorMesh.read_UBC(mesh_path)

    # === Read merged npy ===
    data = np.load(cleaned_npy_path)
    diff_vals = data[:, 3]   # 4th column: diff
    den_vals  = data[:, 4]   # 5th column: density
    susc_vals = data[:, 5]   # 6th column: susceptibility

    # === Define output paths ===
    output_dir = Path(output_dir)
    diff_out  = output_dir / "diff_model_restored.txt"
    dens_out  = output_dir / "density_model_restored.txt"
    susc_out  = output_dir / "susc_model_restored.txt"

    # === Use SimPEG/discretize to write models in UBC format ===
    mesh.write_model_UBC(diff_out, diff_vals)
    mesh.write_model_UBC(dens_out, den_vals)
    mesh.write_model_UBC(susc_out, susc_vals)

    print("Exported as UBC format (readable by meshtool):")
    print(f"Diff:    {diff_out}")
    print(f"Density: {dens_out}")
    print(f"Susc:    {susc_out}")


# === Run ===
if __name__ == "__main__":
    split_and_save_with_simpeg(
        cleaned_npy_path=r"C:\Users\sunlo\Desktop\Research\Modified_Geological_Hydrogen\Data_Cut\merged_model_cleaned.npy",
        mesh_path=r"C:\Users\sunlo\Desktop\Research\Modified_Geological_Hydrogen\mesh_joint_Hannah_exfine_rm.txt",
        output_dir=r"C:\Users\sunlo\Desktop\Research\Modified_Geological_Hydrogen\Data_Cut"
    )
