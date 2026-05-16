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
from pathlib import Path

import h5py
import numpy as np
import yaml

from gprmax_gui_pyside6 import BuildArtifacts
from gprmax_gui_pyside6 import GprMaxRunner
from gprmax_gui_pyside6 import SimulationConfig


class TestUavGprHandoffContract(unittest.TestCase):

    def write_merged_output(self, filename):
        with h5py.File(filename, "w") as fobj:
            fobj.attrs["Title"] = "handoff contract test"
            fobj.attrs["gprMax"] = "test"
            fobj.attrs["Iterations"] = 3
            fobj.attrs["dt"] = 1e-10
            fobj.attrs["nrx"] = 1
            fobj.attrs["nsrc"] = 1
            fobj.attrs["nx_ny_nz"] = (10, 10, 1)
            fobj.attrs["dx_dy_dz"] = (0.01, 0.01, 0.01)
            fobj.attrs["srcsteps"] = (1, 0, 0)
            fobj.attrs["rxsteps"] = (1, 0, 0)
            fobj.attrs["MergedModelCount"] = 2
            fobj.attrs["MergedModelNumbers"] = np.array([1, 2], dtype=np.int32)
            grp = fobj.create_group("/rxs/rx1")
            grp.attrs["Name"] = "rx"
            grp.attrs["Position"] = (0.10, 0.20, 0.0)
            grp["Ez"] = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
            grp["Positions"] = np.array(
                [[0.10, 0.20, 0.0], [0.11, 0.20, 0.0]], dtype=np.float64
            )

    def test_manifest_describes_primary_merged_output_for_mygpr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            merged_path = os.path.join(tmpdir, "handoff_merged.out")
            manifest_path = os.path.join(tmpdir, "handoff_manifest.json")
            input_path = os.path.join(tmpdir, "handoff.in")
            preview_path = os.path.join(tmpdir, "handoff_preview.png")
            metadata_path = os.path.join(tmpdir, "handoff_metadata.json")
            ground_truth_path = os.path.join(tmpdir, "ground_truth.yaml")
            bscan_path = os.path.join(tmpdir, "handoff_bscan.png")
            self.write_merged_output(merged_path)
            for path in (input_path, preview_path, metadata_path, bscan_path):
                with open(path, "w", encoding="utf-8") as fobj:
                    fobj.write("test\n")

            config = SimulationConfig(
                output_root=tmpdir,
                output_name="handoff",
                scan_step=0.012,
                dx=0.01,
                n_traces=2,
                host_eps_r=4.0,
            )
            artifacts = BuildArtifacts(
                output_dir=tmpdir,
                input_path=input_path,
                preview_path=preview_path,
                metadata_path=metadata_path,
                manifest_path=manifest_path,
                ground_truth_path=ground_truth_path,
                primary_out_path=merged_path,
                merged_out_path=merged_path,
                bscan_png_path=bscan_path,
                output_out_paths=[
                    os.path.join(tmpdir, "handoff1.out"),
                    os.path.join(tmpdir, "handoff2.out"),
                ],
            )

            GprMaxRunner()._write_manifest(config, artifacts)

            with open(manifest_path, "r", encoding="utf-8") as fobj:
                manifest = json.load(fobj)

            self.assertEqual(manifest["schema"], "uavgpr_manifest_v1")
            self.assertEqual(manifest["primary_out_file"], merged_path)
            self.assertEqual(manifest["merged_out_file"], merged_path)
            self.assertEqual(manifest["component"], "Ez")
            self.assertTrue(manifest["raw_is_unchanged"])
            self.assertTrue(manifest["preview_processing_only"])
            self.assertIn("scan_geometry", manifest)
            self.assertIn("medium", manifest)
            self.assertIn("simple_targets", manifest)
            self.assertEqual(manifest["requested_scan_step_m"], 0.012)
            self.assertEqual(manifest["scan_step_cells"], 1)
            self.assertEqual(manifest["effective_scan_step_m"], 0.01)
            self.assertEqual(
                manifest["paths_relative_to_output_dir"]["primary_out_file"],
                "handoff_merged.out",
            )
            self.assertEqual(
                manifest["paths_relative_to_output_dir"]["input_file"],
                "handoff.in",
            )
            readiness = manifest["dataset_readiness"]
            self.assertTrue(readiness["input_file"])
            self.assertTrue(readiness["primary_out_file"])
            self.assertTrue(readiness["merged_out_file"])
            self.assertTrue(readiness["metadata_file"])
            self.assertTrue(readiness["ground_truth_file"])
            self.assertTrue(readiness["bscan_preview_file"])
            self.assertEqual(
                manifest["paths_relative_to_output_dir"]["ground_truth_file"],
                "ground_truth.yaml",
            )
            self.assertEqual(
                manifest["gprmax_notes"]["preferred_bscan_dataset"], "/rxs/rx1/Ez"
            )
            self.assertEqual(
                manifest["gprmax_notes"]["data_layout"],
                "samples x traces; single A-scan is stored as samples x 1 by GUI readers",
            )

            summary = manifest["primary_out_summary"]
            self.assertTrue(summary["exists"])
            self.assertEqual(summary["attrs"]["MergedModelCount"], 2)
            self.assertEqual(summary["attrs"]["MergedModelNumbers"], [1, 2])
            self.assertEqual(summary["attrs"]["dx_dy_dz"], [0.01, 0.01, 0.01])
            self.assertEqual(summary["attrs"]["rxsteps"], [1, 0, 0])
            self.assertEqual(summary["receivers"]["rx1"]["datasets"]["Ez"], [3, 2])
            self.assertEqual(
                summary["receivers"]["rx1"]["datasets"]["Positions"], [2, 3]
            )

            with open(ground_truth_path, "r", encoding="utf-8") as fobj:
                ground_truth = yaml.safe_load(fobj)

            schema = json.loads(
                Path("docs/schemas/gprmax_ground_truth.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            try:
                import jsonschema
            except Exception:
                jsonschema = None
            if jsonschema is not None:
                jsonschema.validate(ground_truth, schema)

            self.assertEqual(ground_truth["schema"], "gprmax_ground_truth_v1")
            self.assertEqual(ground_truth["dataset_id"], "handoff")
            self.assertEqual(ground_truth["model_file"], "handoff.in")
            self.assertEqual(ground_truth["output_file"], "handoff_merged.out")
            self.assertEqual(ground_truth["target"]["type"], "pipe")
            self.assertEqual(ground_truth["target"]["material"], "pec")
            self.assertAlmostEqual(ground_truth["target"]["depth_m"], 0.09)
            self.assertIn("trace_range", ground_truth["target_roi"])
            self.assertIn("sample_range", ground_truth["target_roi"])
            self.assertIn("trace_range", ground_truth["background_roi"])
            self.assertIn("sample_range", ground_truth["background_roi"])
            self.assertIn("cnr_db", ground_truth["metrics"])
            self.assertIsNone(ground_truth["metrics"]["cnr_db"])


if __name__ == "__main__":
    unittest.main()
