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
import unittest

from gprmax_gui_pyside6 import PRESETS
from gprmax_gui_pyside6 import PhysicsAuditor
from gprmax_gui_pyside6 import ScenarioBuilder
from gprmax_gui_pyside6 import build_smoke_config


class TestUavGprRealisticPipePreset(unittest.TestCase):

    def make_args(self, traces):
        return argparse.Namespace(
            output_root=".",
            python=None,
            gpu=False,
            traces=traces,
            smoke_preset="realistic_pipe_bscan",
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


if __name__ == "__main__":
    unittest.main()
