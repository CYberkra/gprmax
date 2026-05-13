#!/usr/bin/env python3
"""Convert gprMax HDF5 output to CSV format for GPR_GUI"""

import os
import glob
import h5py
import numpy as np
import pandas as pd


def convert_gprmax_to_csv(input_dir, output_dir):
    """Convert gprMax output files to CSV format."""

    os.makedirs(output_dir, exist_ok=True)

    output_files = sorted(glob.glob(os.path.join(input_dir, "mountain_peak*.out")))

    print(f"Found {len(output_files)} output files")

    for idx, filepath in enumerate(output_files, 1):
        print(f"Processing {os.path.basename(filepath)}...")

        with h5py.File(filepath, "r") as f:
            dt = f.attrs["dt"]
            iterations = f.attrs["Iterations"]
            time = np.linspace(0, (iterations - 1) * dt, num=iterations)

            rx_path = "/rxs/rx1/"
            available_outputs = list(f[rx_path].keys())
            print(f"  Available outputs: {available_outputs}")

            output_data = f[rx_path + "Ez"][:]

            df = pd.DataFrame({"time_s": time, "amplitude": output_data})

            output_csv = os.path.join(output_dir, f"lineData_{idx:07d}.csv")
            df.to_csv(output_csv, index=False, header=False)
            print(f"  Saved to {output_csv}")

    print(
        f"\nConversion complete! {len(output_files)} CSV files created in {output_dir}"
    )


if __name__ == "__main__":
    input_dir = r"E:\gprMax\gprMax-v.3.1.7\user_models"
    output_dir = r"D:\ClawX-Data\sim\gprmax_outcsv"

    convert_gprmax_to_csv(input_dir, output_dir)
