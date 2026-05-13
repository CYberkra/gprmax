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

from gprMax.exceptions import CmdInputError
from tools.outputfiles_merge import get_output_data
from tools.outputfiles_merge import merge_files


class TestOutputfilesMerge(unittest.TestCase):

    def write_output_file(self, filename, values, position):
        with h5py.File(filename, 'w') as f:
            f.attrs['Title'] = 'merge test'
            f.attrs['gprMax'] = 'test'
            f.attrs['Iterations'] = len(values)
            f.attrs['dt'] = 1e-10
            f.attrs['nrx'] = 1
            f.attrs['nsrc'] = 1
            f.attrs['nx_ny_nz'] = (10, 10, 1)
            f.attrs['dx_dy_dz'] = (0.01, 0.01, 0.01)
            f.attrs['srcsteps'] = (0.01, 0, 0)
            f.attrs['rxsteps'] = (0.01, 0, 0)
            grp = f.create_group('/rxs/rx1')
            grp.attrs['Name'] = 'rx'
            grp.attrs['Position'] = position
            grp['Ez'] = np.array(values, dtype=np.float32)

    def test_get_output_data_no_receivers_raises_cmd_input_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'empty.out')
            with h5py.File(filename, 'w') as f:
                f.attrs['nrx'] = 0
                f.attrs['dt'] = 1e-10

            with self.assertRaises(CmdInputError):
                get_output_data(filename, 1, 'Ez')

    def test_merge_files_no_inputs_raises_cmd_input_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(CmdInputError):
                merge_files(os.path.join(tmpdir, 'missing'))

    def test_merge_files_writes_bscan_and_trace_positions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            basefilename = os.path.join(tmpdir, 'model')
            self.write_output_file(basefilename + '1.out', [1, 2, 3], (0.1, 0.2, 0.0))
            self.write_output_file(basefilename + '2.out', [4, 5, 6], (0.2, 0.2, 0.0))

            merge_files(basefilename)

            with h5py.File(basefilename + '_merged.out', 'r') as f:
                np.testing.assert_array_equal(f['/rxs/rx1/Ez'][:], np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32))
                np.testing.assert_array_equal(f['/rxs/rx1/Positions'][:], np.array([[0.1, 0.2, 0.0], [0.2, 0.2, 0.0]]))
                np.testing.assert_array_equal(f.attrs['MergedModelNumbers'], np.array([1, 2], dtype=np.int32))
                self.assertEqual(f.attrs['MergedModelCount'], 2)
                self.assertIn('dx_dy_dz', f.attrs)
                self.assertIn('rxsteps', f.attrs)


if __name__ == '__main__':
    unittest.main()
