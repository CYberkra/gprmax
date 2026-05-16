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

import h5py
import numpy as np

from gprmax_gui_pyside6 import GprMaxRunner
from gprmax_gui_pyside6 import create_bscan_figure


class TestUavGprPreviewAxes(unittest.TestCase):

    def test_loader_returns_receiver_x_axis_from_positions(self):
        with tempfile.TemporaryDirectory() as root:
            out_path = os.path.join(root, "case_merged.out")
            with h5py.File(out_path, "w") as fobj:
                fobj.attrs["dt"] = 1e-10
                rxgroup = fobj.create_group("/rxs/rx1")
                rxgroup["Ez"] = np.array([[1, 2], [3, 4]], dtype=np.float32)
                rxgroup["Positions"] = np.array(
                    [[0.22, 0.70, 0.0], [0.23, 0.70, 0.0]],
                    dtype=np.float64,
                )

            data, dt, x_axis = GprMaxRunner().load_bscan_with_axes(out_path)

            self.assertEqual(data.shape, (2, 2))
            self.assertEqual(dt, 1e-10)
            np.testing.assert_allclose(x_axis, [0.22, 0.23])

    def test_bscan_figure_uses_distance_axis_when_available(self):
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)

        figure = create_bscan_figure(data, 1e-10, x_axis_m=np.array([0.22, 0.23]))
        ax = figure.axes[0]

        self.assertEqual(ax.get_xlabel(), "距离 x [m]")
        self.assertAlmostEqual(ax.images[0].get_extent()[0], 0.22)
        self.assertAlmostEqual(ax.images[0].get_extent()[1], 0.23)


if __name__ == "__main__":
    unittest.main()
