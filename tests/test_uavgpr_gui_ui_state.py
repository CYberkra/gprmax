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
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from gprmax_gui_pyside6 import MainWindow


class TestUavGprGuiUiState(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_default_gui_opens_on_uavgpr_baseline_with_simple_frontend(self):
        window = MainWindow()
        try:
            config = window.read_config()

            self.assertEqual(config.preset_key, "uav_pipe_gain_workflow_bscan")
            self.assertEqual(window.output_name_edit.text(), "uavgpr_baseline")
            self.assertFalse(window.advanced_options_check.isChecked())

            self.assertTrue(window.source_group.isHidden())
            self.assertTrue(window.domain_x_spin.isHidden())
            self.assertTrue(window.dx_spin.isHidden())
            self.assertTrue(window.python_edit.isHidden())
            self.assertTrue(window.geometry_only_check.isHidden())

            self.assertFalse(window.ground_surface_y_spin.isHidden())
            self.assertFalse(window.lift_off_spin.isHidden())
            self.assertFalse(window.host_eps_spin.isHidden())
            self.assertFalse(window.host_sigma_spin.isHidden())
            self.assertFalse(window.processing_report_button.isHidden())
        finally:
            window.close()

    def test_advanced_toggle_restores_low_level_controls(self):
        window = MainWindow()
        try:
            window.advanced_options_check.setChecked(True)

            self.assertFalse(window.source_group.isHidden())
            self.assertFalse(window.domain_x_spin.isHidden())
            self.assertFalse(window.dx_spin.isHidden())
            self.assertFalse(window.python_edit.isHidden())
            self.assertFalse(window.geometry_only_check.isHidden())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
