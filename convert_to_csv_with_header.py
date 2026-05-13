#!/usr/bin/env python3
"""Convert gprMax HDF5 output to CSV format for GPR_GUI with proper headers"""

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

    first_file = output_files[0]
    with h5py.File(first_file, "r") as f:
        dt = f.attrs["dt"]
        iterations = f.attrs["Iterations"]

    time_step_ns = dt * 1e9
    total_time_ns = dt * iterations * 1e9

    num_samples = iterations
    num_traces = len(output_files)
    trace_interval_m = 0.01

    header = f"""# Number of Samples = {num_samples}
# Time windows = {total_time_ns}
# Number of Traces = {num_traces}
# Trace interval = {trace_interval_m}"""

    print(f"Header info:")
    print(f"  Number of Samples: {num_samples}")
    print(f"  Time windows: {total_time_ns} ns")
    print(f"  Number of Traces: {num_traces}")
    print(f"  Trace interval: {trace_interval_m} m")
    print(f"  Time step: {time_step_ns} ns")

    for idx, filepath in enumerate(output_files, 1):
        print(f"Processing {os.path.basename(filepath)}...")

        with h5py.File(filepath, "r") as f:
            time = np.linspace(0, (iterations - 1) * dt, num=iterations)
            output_data = f["/rxs/rx1/Ez"][:]

            csv_content = header + "\n"
            for t, amp in zip(time, output_data):
                csv_content += f"{t},{amp}\n"

            output_csv = os.path.join(output_dir, f"lineData_{idx:07d}.csv")
            with open(output_csv, "w") as f:
                f.write(csv_content)
            print(f"  Saved to {output_csv}")

    print(
        f"\nConversion complete! {len(output_files)} CSV files created in {output_dir}"
    )


if __name__ == "__main__":
    input_dir = r"E:\gprMax\gprMax-v.3.1.7\user_models"
    output_dir = r"D:\ClawX-Data\sim\gprmax_outcsv"

    convert_gprmax_to_csv(input_dir, output_dir)
