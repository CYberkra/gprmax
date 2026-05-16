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
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from uavgpr_processing_report import run_processing_report


MYGPR_ROOT = Path(r"D:\MyGPR")


@unittest.skipUnless(MYGPR_ROOT.exists(), "MyGPR checkout is required")
class TestUavGprProcessingReport(unittest.TestCase):

    def test_processing_report_runs_mygpr_combinations_without_touching_raw(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_path = root / "uav_case_merged.out"
            manifest_path = root / "uav_case_manifest.json"
            raw = self._make_bscan()
            self._write_out(out_path, raw)
            manifest_path.write_text(
                json.dumps(
                    {
                        "primary_out_file": str(out_path),
                        "component": "Ez",
                        "effective_scan_step_m": 0.01,
                        "uav_lift_off_m": 0.15,
                        "raw_is_unchanged": True,
                        "preview_processing_only": True,
                        "simple_targets": [
                            {
                                "enabled": True,
                                "shape": "cylinder",
                                "material_name": "pec",
                                "center_x_m": 0.62,
                                "center_y_m": 0.22,
                                "radius_m": 0.035,
                            }
                        ],
                        "scan_geometry": {
                            "source_start_x_m": 0.10,
                            "receiver_offset_m": 0.12,
                            "effective_scan_step_m": 0.01,
                            "ground_surface_y_m": 0.55,
                            "uav_lift_off_m": 0.15,
                            "scan_mid_x_m": 0.62,
                        },
                        "medium": {
                            "host_name": "dry_soil",
                            "host_eps_r": 9.0,
                            "host_sigma": 0.004,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            summary = run_processing_report(
                str(manifest_path), mygpr_root=str(MYGPR_ROOT)
            )

            self.assertTrue(Path(summary["report_html"]).exists())
            self.assertTrue(Path(summary["summary_json"]).exists())
            self.assertEqual(summary["data_shape"], [raw.shape[0], raw.shape[1]])
            self.assertTrue(summary["raw_is_unchanged"])
            self.assertTrue(summary["preview_processing_only"])
            self.assertEqual(summary["mygpr_bridge_schema"], "mygpr_gprmax_report_bridge_v1")
            self.assertEqual(len(summary["target_rois"]), 1)
            self.assertEqual(len(summary["candidates"]), 15)
            candidate_ids = {item["candidate_id"] for item in summary["candidates"]}
            self.assertIn("subtracting_average_2D__time_power_gain", candidate_ids)
            self.assertIn("median_background_2D__agcGain", candidate_ids)
            self.assertTrue(summary["recommended"]["candidate_id"])

            with h5py.File(str(out_path), "r") as fobj:
                np.testing.assert_array_equal(raw, fobj["/rxs/rx1/Ez"][:])

    def test_processing_report_warns_when_trace_count_is_too_small_to_rank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_path = root / "tiny_merged.out"
            manifest_path = root / "tiny_manifest.json"
            raw = self._make_bscan()[:, :2]
            self._write_out(out_path, raw)
            self._write_manifest(manifest_path, out_path)

            summary = run_processing_report(
                str(manifest_path), mygpr_root=str(MYGPR_ROOT)
            )

            codes = [item["code"] for item in summary["warnings"]]
            self.assertIn("insufficient_trace_count_for_ranking", codes)

    def test_processing_report_includes_roi_for_each_simple_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_path = root / "multi_merged.out"
            manifest_path = root / "multi_manifest.json"
            raw = self._make_bscan()
            self._write_out(out_path, raw)
            self._write_manifest(
                manifest_path,
                out_path,
                targets=[
                    {
                        "enabled": True,
                        "shape": "cylinder",
                        "material_name": "pec",
                        "center_x_m": 0.42,
                        "center_y_m": 0.22,
                        "radius_m": 0.035,
                    },
                    {
                        "enabled": True,
                        "shape": "box",
                        "material_name": "target_concrete",
                        "center_x_m": 0.72,
                        "center_y_m": 0.28,
                        "width_m": 0.050,
                        "height_m": 0.040,
                    },
                ],
            )

            summary = run_processing_report(
                str(manifest_path), mygpr_root=str(MYGPR_ROOT)
            )

            self.assertEqual(len(summary["target_rois"]), 2)

    def _make_bscan(self):
        samples = 96
        traces = 32
        time = np.arange(samples, dtype=np.float32)[:, np.newaxis]
        trace = np.arange(traces, dtype=np.float32)[np.newaxis, :]
        direct = 0.12 * np.sin(0.8 * time) * np.exp(-time / 80.0)
        background = np.repeat(direct, traces, axis=1)
        center = 16.0
        hyperbola = 58.0 + 0.12 * (trace - center) ** 2
        target = np.exp(-((time - hyperbola) ** 2) / 18.0)
        ringing = 0.03 * np.sin(1.7 * time + 0.3 * trace)
        return (background + 0.08 * target + ringing).astype(np.float32)

    def _write_out(self, path, data):
        with h5py.File(str(path), "w") as fobj:
            fobj.attrs["dt"] = 1.0e-10
            fobj.attrs["Iterations"] = data.shape[0]
            rxgroup = fobj.create_group("rxs").create_group("rx1")
            rxgroup.create_dataset("Ez", data=data)
            rxgroup.create_dataset(
                "Positions",
                data=np.column_stack(
                    [
                        np.linspace(0.22, 0.53, data.shape[1]),
                        np.full(data.shape[1], 0.70),
                        np.zeros(data.shape[1]),
                    ]
                ),
            )

    def _write_manifest(self, manifest_path, out_path, targets=None):
        manifest_path.write_text(
            json.dumps(
                {
                    "primary_out_file": str(out_path),
                    "component": "Ez",
                    "effective_scan_step_m": 0.01,
                    "uav_lift_off_m": 0.15,
                    "raw_is_unchanged": True,
                    "preview_processing_only": True,
                    "simple_targets": targets
                    or [
                        {
                            "enabled": True,
                            "shape": "cylinder",
                            "material_name": "pec",
                            "center_x_m": 0.62,
                            "center_y_m": 0.22,
                            "radius_m": 0.035,
                        }
                    ],
                    "scan_geometry": {
                        "source_start_x_m": 0.10,
                        "receiver_offset_m": 0.12,
                        "effective_scan_step_m": 0.01,
                        "ground_surface_y_m": 0.55,
                        "uav_lift_off_m": 0.15,
                        "scan_mid_x_m": 0.62,
                    },
                    "medium": {
                        "host_name": "dry_soil",
                        "host_eps_r": 9.0,
                        "host_sigma": 0.004,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
