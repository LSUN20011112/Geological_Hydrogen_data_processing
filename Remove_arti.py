import numpy as np
import re
import ast

# === input paths ===
merged_npy_path  = r"C:\Users\sunlo\Desktop\Research\Modified_Geological_Hydrogen\merged_model.npy"
cut_plan_path    = r"C:\Users\sunlo\Desktop\Research\Modified_Geological_Hydrogen\Data_Cut\Cut_plan.txt"
output_npy_path = r"C:\Users\sunlo\Desktop\Research\Modified_Geological_Hydrogen\Data_Cut\merged_model_cleaned.npy"
output_xyz_path = r"C:\Users\sunlo\Desktop\Research\Modified_Geological_Hydrogen\Data_Cut\merged_model_cleaned.xyz"

def read_cut_boxes(path):
    """Read 'Unit k:' then '(xmin,xmax,ymin,ymax,zmin,zmax)' into a dict {k: (xmin,xmax,ymin,ymax,zmin,zmax)}."""
    boxes = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    cur = None
    for ln in lines:
        if ln.lower().startswith("unit"):
            m = re.search(r"unit\s+(\d+)", ln, flags=re.IGNORECASE)
            if m:
                cur = int(m.group(1))
        elif ln.startswith("(") and cur is not None:
            t = ast.literal_eval(ln)
            if not (isinstance(t, tuple) and len(t) == 6):
                raise ValueError(f"Bad box for Unit {cur}: {ln}")
            xmin, xmax, ymin, ymax, zmin, zmax = t
            xmin, xmax = sorted((float(xmin), float(xmax)))
            ymin, ymax = sorted((float(ymin), float(ymax)))
            zmin, zmax = sorted((float(zmin), float(zmax)))
            boxes[cur] = (xmin, xmax, ymin, ymax, zmin, zmax)
            cur = None
    if not boxes:
        raise ValueError("No unit boxes parsed from cut plan.")
    return boxes

def clean_unit_by_unit(merged_path, cut_path, out_xyz, out_npy):
    # load merged (N,6): X Y Z Diff Density Susc
    data = np.load(merged_path)
    assert data.shape[1] == 6, "Merged .npy must be (N,6)"
    X, Y, Z = data[:,0], data[:,1], data[:,2]
    diff    = data[:,3]
    dens    = data[:,4]
    susc    = data[:,5]

    # robust integer unit labels from column 4
    diff_int = np.rint(diff).astype(int)

    # read cut boxes
    boxes = read_cut_boxes(cut_path)

    # start from original values; we will zero-out outside-box points per unit
    cleaned = data.copy()

    # iterate units in the cut plan, one by one
    for unit_id in sorted(boxes.keys()):
        if unit_id == 0:
            # background unit doesn't need a box-cut; skip
            continue

        xmin, xmax, ymin, ymax, zmin, zmax = boxes[unit_id]

        # select ONLY points of THIS unit (from column 4)
        is_this_unit = (diff_int == unit_id)

        # among these points, check box condition
        in_box = (
            (X >= xmin) & (X <= xmax) &
            (Y >= ymin) & (Y <= ymax) &
            (Z >= zmin) & (Z <= zmax)
        )

        # outside = this unit's points that are NOT in its own box
        outside = is_this_unit & (~in_box)

        # set to background (0,0,0) for outside
        cleaned[outside, 3] = 0.0  # diff -> 0
        cleaned[outside, 4] = 0.0  # dens -> 0
        cleaned[outside, 5] = 0.0  # susc -> 0

        # optional: quick stats
        kept   = int((is_this_unit & in_box).sum())
        removed= int(outside.sum())
        total  = int(is_this_unit.sum())
        print(f"Unit {unit_id}: total={total}, kept={kept}, removed={removed}")

    # final safety: ensure any diff==0 rows carry (0,0) for dens/susc
    bg = (np.rint(cleaned[:,3]).astype(int) == 0)
    cleaned[bg, 4] = 0.0
    cleaned[bg, 5] = 0.0

    # save
    np.savetxt(out_xyz, cleaned, fmt="%.6f", header="X Y Z Diff Density Susc")
    np.save(out_npy, cleaned)
    print(f"Saved:\n  XYZ: {out_xyz}\n  NPY: {out_npy}")

# run
clean_unit_by_unit(merged_npy_path, cut_plan_path, output_xyz_path, output_npy_path)
