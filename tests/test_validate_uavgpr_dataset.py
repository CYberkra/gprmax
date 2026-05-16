# Copyright (C) 2015-2023: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

import json
import os
import tempfile
import unittest

import h5py
import numpy as np
import yaml

from scripts.validate_uavgpr_dataset import main as validator_main
from scripts.validate_uavgpr_dataset import validate_dataset


class TestValidateUavGprDataset(unittest.TestCase):

    def write_dataset(self, root, positions=None, summary_positions_shape=None):
        input_path = os.path.join(root, "case.in")
        out_path = os.path.join(root, "case_merged.out")
        metadata_path = os.path.join(root, "case_metadata.json")
        manifest_path = os.path.join(root, "case_manifest.json")
        ground_truth_path = os.path.join(root, "ground_truth.yaml")

        with open(input_path, "w", encoding="utf-8") as fobj:
            fobj.write("#title: validator test\n")

        if positions is None:
            positions = np.array(
                [[0.220, 0.540, 0.0], [0.230, 0.540, 0.0]], dtype=np.float64
            )

        with h5py.File(out_path, "w") as fobj:
            fobj.attrs["Title"] = "validator test"
            fobj.attrs["Iterations"] = 3
            fobj.attrs["dt"] = 1e-10
            fobj.attrs["nrx"] = 1
            fobj.attrs["nsrc"] = 1
            fobj.attrs["dx_dy_dz"] = (0.01, 0.01, 0.01)
            fobj.attrs["rxsteps"] = (1, 0, 0)
            fobj.attrs["MergedModelCount"] = 2
            grp = fobj.create_group("/rxs/rx1")
            grp.attrs["Position"] = (0.220, 0.540, 0.0)
            grp["Ez"] = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
            grp["Positions"] = positions

        metadata = {
            "config": {
                "target_shape": "cylinder",
                "target_center_x": 0.175,
                "target_radius": 0.010,
                "source_start_x": 0.120,
                "receiver_offset": 0.100,
                "n_traces": 2,
                "effective_scan_step": 0.010,
            }
        }
        with open(metadata_path, "w", encoding="utf-8") as fobj:
            json.dump(metadata, fobj)

        ground_truth = {
            "schema": "gprmax_ground_truth_v1",
            "dataset_id": "case",
            "model_file": "case.in",
            "output_file": "case_merged.out",
            "target_roi": {
                "trace_range": [0, 1],
                "sample_range": [0, 2],
            },
            "background_roi": {
                "trace_range": [0, 1],
                "sample_range": [0, 2],
            },
            "target": {
                "type": "pipe",
                "depth_m": 0.10,
                "material": "pec",
                "center_x_m": 0.175,
                "center_y_m": 0.080,
                "radius_m": 0.010,
            },
            "metadata": {
                "auto_generated": True,
                "roi_review_required": True,
                "roi_method": "geometric_twt_estimate",
            },
            "metrics": {
                "cnr_db": None,
                "background_energy": None,
                "target_energy": None,
                "localization_error_trace": None,
                "localization_error_sample": None,
            },
        }
        with open(ground_truth_path, "w", encoding="utf-8") as fobj:
            yaml.safe_dump(ground_truth, fobj, sort_keys=False)

        if summary_positions_shape is None:
            summary_positions_shape = [2, 3]
        manifest = {
            "schema": "uavgpr_manifest_v1",
            "input_file": input_path,
            "metadata_file": metadata_path,
            "primary_out_file": out_path,
            "component": "Ez",
            "dataset_readiness": {
                "input_file": True,
                "primary_out_file": True,
                "merged_out_file": True,
                "metadata_file": True,
                "ground_truth_file": True,
                "bscan_preview_file": False,
                "background_preview_files": False,
            },
            "paths_relative_to_output_dir": {
                "ground_truth_file": "ground_truth.yaml",
            },
            "primary_out_summary": {
                "attrs": {
                    "Iterations": 3,
                    "dt": 1e-10,
                    "nrx": 1,
                    "nsrc": 1,
                    "dx_dy_dz": [0.01, 0.01, 0.01],
                    "rxsteps": [1, 0, 0],
                },
                "receivers": {
                    "rx1": {
                        "datasets": {
                            "Ez": [3, 2],
                            "Positions": summary_positions_shape,
                        }
                    }
                },
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as fobj:
            json.dump(manifest, fobj)
        return manifest_path

    def read_ground_truth(self, root):
        with open(os.path.join(root, "ground_truth.yaml"), "r", encoding="utf-8") as fobj:
            return yaml.safe_load(fobj)

    def write_ground_truth(self, root, ground_truth):
        with open(os.path.join(root, "ground_truth.yaml"), "w", encoding="utf-8") as fobj:
            yaml.safe_dump(ground_truth, fobj, sort_keys=False)

    def test_valid_dataset_passes_contract_checks(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_dataset(root)

            result = validate_dataset(root)

            self.assertFalse(
                result.has_errors(),
                "\n".join(message.text for message in result.errors()),
            )

    def test_missing_ground_truth_is_reported_when_readiness_is_true(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_dataset(root)
            os.remove(os.path.join(root, "ground_truth.yaml"))

            result = validate_dataset(root)

            self.assertTrue(result.has_errors())
            self.assertIn(
                "ground_truth.yaml is missing",
                "\n".join(message.text for message in result.errors()),
            )
            self.assertEqual(validator_main([root]), 1)

    def test_ground_truth_roi_out_of_bounds_is_reported(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_dataset(root)
            ground_truth = self.read_ground_truth(root)
            ground_truth["target_roi"]["sample_range"] = [0, 3]
            self.write_ground_truth(root, ground_truth)

            result = validate_dataset(root)

            self.assertTrue(result.has_errors())
            self.assertIn(
                "ground_truth target_roi.sample_range exceeds Ez sample count 3",
                "\n".join(message.text for message in result.errors()),
            )

    def test_ground_truth_output_file_must_match_primary_out(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_dataset(root)
            ground_truth = self.read_ground_truth(root)
            ground_truth["output_file"] = "other_merged.out"
            self.write_ground_truth(root, ground_truth)

            result = validate_dataset(root)

            self.assertTrue(result.has_errors())
            self.assertIn(
                "ground_truth output_file does not match manifest primary_out_file",
                "\n".join(message.text for message in result.errors()),
            )

    def test_ground_truth_missing_schema_field_is_reported(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_dataset(root)
            ground_truth = self.read_ground_truth(root)
            ground_truth["metrics"].pop("cnr_db")
            self.write_ground_truth(root, ground_truth)

            result = validate_dataset(root)

            self.assertTrue(result.has_errors())
            self.assertIn(
                "ground_truth",
                "\n".join(message.text for message in result.errors()),
            )
            self.assertIn(
                "cnr_db",
                "\n".join(message.text for message in result.errors()),
            )

    def test_position_step_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as root:
            bad_positions = np.array(
                [[0.220, 0.540, 0.0], [0.235, 0.540, 0.0]], dtype=np.float64
            )
            self.write_dataset(root, positions=bad_positions)

            result = validate_dataset(root)

            self.assertTrue(result.has_errors())
            self.assertIn(
                "Positions do not match rxsteps * dx_dy_dz",
                "\n".join(message.text for message in result.errors()),
            )

    def test_manifest_shape_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_dataset(root, summary_positions_shape=[3, 3])

            result = validate_dataset(root)

            self.assertTrue(result.has_errors())
            self.assertIn(
                "manifest dataset Positions shape does not match HDF5",
                "\n".join(message.text for message in result.errors()),
            )

    def test_missing_manifest_schema_is_reported(self):
        with tempfile.TemporaryDirectory() as root:
            manifest_path = self.write_dataset(root)
            with open(manifest_path, "r", encoding="utf-8") as fobj:
                manifest = json.load(fobj)
            manifest.pop("schema")
            with open(manifest_path, "w", encoding="utf-8") as fobj:
                json.dump(manifest, fobj)

            result = validate_dataset(root)

            self.assertTrue(result.has_errors())
            self.assertIn(
                "manifest schema must be uavgpr_manifest_v1",
                "\n".join(message.text for message in result.errors()),
            )

    def test_readiness_primary_out_file_is_required(self):
        with tempfile.TemporaryDirectory() as root:
            manifest_path = self.write_dataset(root)
            with open(manifest_path, "r", encoding="utf-8") as fobj:
                manifest = json.load(fobj)
            manifest["dataset_readiness"]["primary_out_file"] = False
            with open(manifest_path, "w", encoding="utf-8") as fobj:
                json.dump(manifest, fobj)

            result = validate_dataset(root)

            self.assertTrue(result.has_errors())
            self.assertIn(
                "dataset_readiness.primary_out_file is not true",
                "\n".join(message.text for message in result.errors()),
            )


if __name__ == "__main__":
    unittest.main()
