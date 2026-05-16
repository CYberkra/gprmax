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

import argparse
import math
import unittest

import numpy as np

from gprmax_gui_pyside6 import PRESETS
from gprmax_gui_pyside6 import PhysicsAuditor
from gprmax_gui_pyside6 import ScenarioBuilder
from gprmax_gui_pyside6 import SimpleTargetSpec
from gprmax_gui_pyside6 import SimulationConfig
from gprmax_gui_pyside6 import apply_mild_time_gain
from gprmax_gui_pyside6 import build_smoke_config
from gprmax_gui_pyside6 import material_velocity
from gprmax_gui_pyside6 import remove_horizontal_background


class TestUavGprRealisticPipePreset(unittest.TestCase):

    def make_args(self, traces):
        return self.make_args_for_preset("realistic_pipe_bscan", traces)

    def make_args_for_preset(self, preset_key, traces):
        return argparse.Namespace(
            output_root=".",
            python=None,
            gpu=False,
            traces=traces,
            smoke_preset=preset_key,
        )

    def test_realistic_pipe_preset_is_available_for_gui(self):
        preset = PRESETS["realistic_pipe_bscan"]

        self.assertEqual(preset["label"], "真实感管线 B-scan")
        self.assertEqual(preset["target_shape"], "cylinder")
        self.assertEqual(preset["target_name"], "pec")
        self.assertGreater(preset["time_window_ns"], 10.0)
        self.assertGreater(preset["receiver_offset"], 0.05)

    def test_realistic_pipe_preset_builds_audited_input(self):
        config = build_smoke_config(
            self.make_args(traces=PRESETS["realistic_pipe_bscan"]["n_traces"])
        )
        report = PhysicsAuditor().build_report(config)

        self.assertFalse(
            report.has_errors(),
            "\n".join(
                message.text for message in report.messages if message.level == "error"
            ),
        )
        self.assertLessEqual(config.receiver_end_x, config.domain_x)
        self.assertLess(config.source_start_x, config.target_center_x)
        self.assertGreater(config.source_end_x, config.target_center_x)
        self.assertGreater(config.time_window_ns, report.derived["closest_twt_ns"])

        input_text = ScenarioBuilder().build_input_text(config)
        self.assertIn("#material: 9 0.001 1 0 dry_soil", input_text)
        self.assertIn("#hertzian_dipole: z 0.120 0.540 0 my_wave", input_text)
        self.assertIn("#rx: 0.220 0.540 0", input_text)
        self.assertIn("#src_steps: 0.010 0 0", input_text)
        self.assertIn("#rx_steps: 0.010 0 0", input_text)
        self.assertIn("#box: 0 0 0 1.200 0.520 0.005 dry_soil", input_text)
        self.assertIn("#cylinder: 0.620 0.260 0", input_text)

    def test_uav_pipe_gain_workflow_preset_models_raw_processing_need(self):
        preset = PRESETS["uav_pipe_gain_workflow_bscan"]
        config = build_smoke_config(
            self.make_args_for_preset(
                "uav_pipe_gain_workflow_bscan",
                preset["n_traces"],
            )
        )
        report = PhysicsAuditor().build_report(config)

        self.assertFalse(
            report.has_errors(),
            "\n".join(
                message.text for message in report.messages if message.level == "error"
            ),
        )
        self.assertEqual(config.lift_off, 0.150)
        self.assertEqual(config.host_sigma, 0.004)
        self.assertEqual(len(config.background_layers), 0)
        self.assertGreater(config.time_window_ns, report.derived["closest_twt_ns"])

        input_text = ScenarioBuilder().build_input_text(config)
        self.assertIn("#material: 9 0.004 1 0 dry_soil", input_text)
        self.assertNotIn("uav_shallow_dry_soil", input_text)
        self.assertNotIn("uav_deeper_moist_soil", input_text)
        self.assertIn("#hertzian_dipole: z 0.100 0.700 0 my_wave", input_text)
        self.assertIn("#rx: 0.220 0.700 0", input_text)
        self.assertIn("#box: 0 0 0 1.200 0.550 0.005 dry_soil", input_text)
        self.assertIn("#cylinder: 0.620 0.220 0", input_text)
        notes = ScenarioBuilder()._gprmax_processing_notes(config)
        self.assertTrue(notes["raw_is_unchanged"])
        self.assertTrue(notes["preview_processing_only"])
        self.assertEqual(notes["scan_geometry"]["uav_lift_off_m"], 0.150)
        self.assertEqual(len(notes["simple_targets"]), 1)

    def test_optional_second_simple_target_is_in_input_and_handoff_notes(self):
        config = build_smoke_config(
            self.make_args_for_preset("uav_pipe_gain_workflow_bscan", 12)
        )
        config.extra_targets = [
            SimpleTargetSpec(
                enabled=True,
                shape="box",
                material_name="target_concrete",
                eps_r=7.0,
                sigma=0.001,
                center_x=0.820,
                center_y=0.260,
                width=0.060,
                height=0.040,
            )
        ]

        input_text = ScenarioBuilder().build_input_text(config)
        self.assertIn("#material: 7 0.001 1 0 target_concrete", input_text)
        self.assertIn("#cylinder: 0.620 0.220 0", input_text)
        self.assertIn("#box: 0.790 0.240 0 0.850 0.280 0.005 target_concrete", input_text)

        notes = ScenarioBuilder()._gprmax_processing_notes(config)
        self.assertEqual(len(notes["simple_targets"]), 2)
        self.assertEqual(notes["simple_targets"][1]["shape"], "box")
        self.assertEqual(notes["simple_targets"][1]["material_name"], "target_concrete")

    def test_extra_target_outside_ground_is_audited(self):
        config = build_smoke_config(
            self.make_args_for_preset("uav_pipe_gain_workflow_bscan", 12)
        )
        config.extra_targets = [
            SimpleTargetSpec(
                enabled=True,
                shape="cylinder",
                material_name="pec",
                center_x=0.820,
                center_y=0.545,
                radius=0.020,
            )
        ]

        report = PhysicsAuditor().build_report(config)

        self.assertTrue(report.has_errors())
        self.assertIn(
            "异常体 #2",
            "\n".join(message.text for message in report.messages),
        )
        self.assertIn(
            "宿主半空间",
            "\n".join(
                message.text for message in report.messages if message.level == "error"
            ),
        )

    def test_closest_twt_uses_effective_scan_step(self):
        config = SimulationConfig(
            domain_x=1.200,
            domain_y=0.850,
            dx=0.010,
            dy=0.010,
            ground_surface_y=0.550,
            lift_off=0.150,
            source_start_x=0.100,
            receiver_offset=0.120,
            scan_step=0.014,
            n_traces=6,
            host_eps_r=9.0,
            target_center_x=0.275,
            target_center_y=0.220,
            target_radius=0.035,
            time_window_ns=50.0,
            center_freq_mhz=500.0,
            preset_key="uav_pipe_gain_workflow_bscan",
        )

        report = PhysicsAuditor().build_report(config)
        velocity = material_velocity(config.host_eps_r)
        expected = min(
            (
                math.hypot(
                    config.source_start_x + index * config.effective_scan_step
                    - config.target_center_x,
                    config.source_y - config.target_center_y,
                )
                + math.hypot(
                    config.source_start_x
                    + index * config.effective_scan_step
                    + config.receiver_offset
                    - config.target_center_x,
                    config.receiver_y - config.target_center_y,
                )
            )
            / velocity
            * 1e9
            for index in range(config.n_traces)
        )

        self.assertEqual(report.derived["effective_scan_step"], config.effective_scan_step)
        self.assertAlmostEqual(report.derived["closest_twt_ns"], expected, places=9)

    def test_preview_processing_keeps_raw_data_separate(self):
        raw = np.array(
            [
                [10.0, 10.0, 10.0],
                [4.0, 6.0, 8.0],
                [1.0, 2.0, 5.0],
            ],
            dtype=np.float32,
        )

        removed = remove_horizontal_background(raw)
        gained = apply_mild_time_gain(removed, 1e-9)

        np.testing.assert_array_equal(raw[0], np.array([10.0, 10.0, 10.0]))
        np.testing.assert_allclose(removed[0], np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(np.mean(removed, axis=1), np.zeros(3), atol=1e-6)
        self.assertGreater(abs(gained[2, 2]), abs(removed[2, 2]))
        np.testing.assert_array_equal(raw[2], np.array([1.0, 2.0, 5.0]))


if __name__ == "__main__":
    unittest.main()
