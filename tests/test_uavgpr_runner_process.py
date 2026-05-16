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

import os
import tempfile
import unittest

from gprmax_gui_pyside6 import BuildArtifacts
from gprmax_gui_pyside6 import GprMaxRunner
from gprmax_gui_pyside6 import SimulationConfig


class FakeProcess(object):
    def __init__(self, lines=None, return_code=0):
        self.stdout = lines or []
        self.return_code = return_code
        self.terminated = False

    def wait(self):
        return self.return_code

    def poll(self):
        return None if not self.terminated else self.return_code

    def terminate(self):
        self.terminated = True


class RecordingRunner(GprMaxRunner):
    def __init__(self):
        super(RecordingRunner, self).__init__()
        self.commands = []
        self.use_gpu_flags = []
        self.fake_process = FakeProcess(["line\n"], 0)

    def _spawn_process(self, command, cwd, use_gpu):
        self.commands.append(list(command))
        self.use_gpu_flags.append(use_gpu)
        return self.fake_process


class TestUavGprRunnerProcess(unittest.TestCase):

    def make_artifacts(self, root):
        return BuildArtifacts(
            output_dir=root,
            input_path=os.path.join(root, "case.in"),
            preview_path=os.path.join(root, "case_preview.png"),
            metadata_path=os.path.join(root, "case_metadata.json"),
        )

    def test_cpu_command_is_default_for_multitrace(self):
        with tempfile.TemporaryDirectory() as root:
            runner = RecordingRunner()
            config = SimulationConfig(
                python_executable="python",
                n_traces=3,
                geometry_fixed=True,
                use_gpu=False,
            )

            runner._run_process(config, self.make_artifacts(root), config.use_gpu)

            command = runner.commands[0]
            self.assertNotIn("-gpu", command)
            self.assertIn("-n", command)
            self.assertIn("3", command)
            self.assertIn("--geometry-fixed", command)
            self.assertEqual(runner.use_gpu_flags, [False])

    def test_gpu_flag_is_only_added_when_requested(self):
        with tempfile.TemporaryDirectory() as root:
            runner = RecordingRunner()
            config = SimulationConfig(
                python_executable="python",
                n_traces=1,
                use_gpu=True,
            )

            runner._run_process(config, self.make_artifacts(root), True)

            self.assertIn("-gpu", runner.commands[0])
            self.assertEqual(runner.use_gpu_flags, [True])

    def test_cancel_terminates_active_process(self):
        runner = RecordingRunner()
        runner.active_process = runner.fake_process

        runner.cancel()

        self.assertTrue(runner.fake_process.terminated)


if __name__ == "__main__":
    unittest.main()
