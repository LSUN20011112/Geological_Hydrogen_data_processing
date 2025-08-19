import numpy as np
import re

# === input path ===
merged_npy_path = r"C:\Users\sunlo\Desktop\Geological_Hydrogen\merged_model_output.npy"
cut_plan_path = r"C:\Users\sunlo\Desktop\Geological_Hydrogen\My_work\Data_Cut\Cut_plan.txt"
output_clean_npy = r"C:\Users\sunlo\Desktop\Geological_Hydrogen\My_work\Data_Cut\merged_model_cleaned.npy"
output_clean_xyz = r"C:\Users\sunlo\Desktop\Geological_Hydrogen\My_work\Data_Cut\merged_model_cleaned.xyz"

# === load merged model ===
points = np.load(merged_npy_path)  # shape = (N, 6)
assert points.shape[1] == 6, "Merged .npy must have shape (N, 6)"

# === read Cut_plan.txt ===
def read_cut_boxes(path):
    unit_boxes = {}
    with open(path) as f:
        lines = f.readlines()

    current_unit = None
    for line in lines:
        line = line.strip()
        if line.startswith("Unit"):
            match = re.search(r"Unit\s+(\d+)", line)
            if match:
                current_unit = int(match.group(1))
        elif line.startswith("(") and current_unit is not None:
            box = eval(line)
            unit_boxes[current_unit] = box
            current_unit = None
    return unit_boxes

unit_boxes = read_cut_boxes(cut_plan_path)

# === cutting ===
cleaned = points.copy()
for unit_id, (x_min, x_max, y_min, y_max, z_min, z_max) in unit_boxes.items():
    # find all the points in a unit
    is_unit = cleaned[:, 3] == unit_id
    # check if the point is in the box
    in_box = (
        (cleaned[:, 0] >= x_min) & (cleaned[:, 0] <= x_max) &
        (cleaned[:, 1] >= y_min) & (cleaned[:, 1] <= y_max) &
        (cleaned[:, 2] >= z_min) & (cleaned[:, 2] <= z_max)
    )
    keep_mask = is_unit & in_box
    remove_mask = is_unit & (~in_box)

    # set it as the value of background
    cleaned[remove_mask, 3:] = 0.0

# === save and output ===
np.save(output_clean_npy, cleaned)
np.savetxt(output_clean_xyz, cleaned, fmt="%.4f", header="X Y Z Diff Density Susc")
print(f"Cleaned model saved:\n→ NPY: {output_clean_npy}\n→ XYZ: {output_clean_xyz}")


