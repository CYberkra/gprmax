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
import glob
import os

import h5py
import numpy as np

from gprMax._version import __version__
from gprMax.exceptions import CmdInputError


def _get_model_output_files(basefilename):
    """Gets numbered output files for a model series."""

    files = glob.glob(basefilename + '[0-9]*.out')
    outputfiles = []
    for filename in files:
        if '_merged' in filename:
            continue
        name = os.path.splitext(filename)[0]
        suffix = name[len(basefilename):]
        if suffix.isdigit():
            outputfiles.append((int(suffix), filename))

    outputfiles.sort(key=lambda item: item[0])
    return outputfiles


def _compare_attr(first, current, name, filename):
    """Checks an HDF5 root attribute is consistent across traces."""

    if name not in first.attrs or name not in current.attrs:
        return

    firstvalue = first.attrs[name]
    currentvalue = current.attrs[name]
    if not np.array_equal(firstvalue, currentvalue):
        raise CmdInputError("Attribute '{}' in {} does not match the first trace".format(name, filename))


def _copy_root_attrs(fin, fout):
    """Copies stable root attributes from an A-scan file to a merged B-scan file."""

    for attr in ['Title', 'Iterations', 'dt', 'nrx', 'nsrc', 'nx_ny_nz', 'dx_dy_dz', 'srcsteps', 'rxsteps']:
        if attr in fin.attrs:
            fout.attrs[attr] = fin.attrs[attr]

    fout.attrs['gprMax'] = __version__


def get_output_data(filename, rxnumber, rxcomponent):
    """Gets B-scan output data from a model.

    Args:
        filename (string): Filename (including path) of output file.
        rxnumber (int): Receiver output number.
        rxcomponent (str): Receiver output field/current component.

    Returns:
        outputdata (array): Array of A-scans, i.e. B-scan data.
        dt (float): Temporal resolution of the model.
    """

    # Open output file and read some attributes
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']
    dt = f.attrs['dt']

    # Check there are any receivers
    if nrx == 0:
        raise CmdInputError('No receivers found in {}'.format(filename))

    path = '/rxs/rx' + str(rxnumber) + '/'
    availableoutputs = list(f[path].keys())

    # Check if requested output is in file
    if rxcomponent not in availableoutputs:
        raise CmdInputError('{} output requested to plot, but the available output for receiver 1 is {}'.format(rxcomponent, ', '.join(availableoutputs)))

    outputdata = f[path + '/' + rxcomponent]
    outputdata = np.array(outputdata)
    f.close()

    return outputdata, dt


def merge_files(basefilename, removefiles=False):
    """Merges traces (A-scans) from multiple output files into one new file,
        then optionally removes the series of output files.

    Args:
        basefilename (string): Base name of output file series including path.
        outputs (boolean): Flag to remove individual output files after merge.
    """

    outputfile = basefilename + '_merged.out'
    outputfiles = _get_model_output_files(basefilename)
    modelruns = len(outputfiles)
    if modelruns == 0:
        raise CmdInputError('No output files found for base filename {}'.format(basefilename))

    modelnumbers = np.array([modelnumber for modelnumber, filename in outputfiles], dtype=np.int32)
    positiondatasets = {}
    outputnames = {}

    # Combined output file
    fout = h5py.File(outputfile, 'w')
    try:
        firstfilename = outputfiles[0][1]
        first = h5py.File(firstfilename, 'r')
        try:
            _copy_root_attrs(first, fout)
            fout.attrs['MergedModelNumbers'] = modelnumbers
            fout.attrs['MergedModelCount'] = modelruns
            iterations = first.attrs['Iterations']
            nrx = first.attrs['nrx']

            for rx in range(1, nrx + 1):
                path = '/rxs/rx' + str(rx)
                if path not in first:
                    raise CmdInputError('Receiver {} not found in {}'.format(rx, firstfilename))
                grp = fout.create_group(path)
                for attr in first[path].attrs:
                    grp.attrs[attr] = first[path].attrs[attr]
                outputnames[rx] = list(first[path].keys())
                if 'Position' in first[path].attrs:
                    positiondatasets[rx] = grp.create_dataset('Positions', (modelruns, 3), dtype=np.float64)
                for output in outputnames[rx]:
                    grp.create_dataset(output, (iterations, modelruns), dtype=first[path + '/' + output].dtype)
        finally:
            first.close()

        for model, (modelnumber, filename) in enumerate(outputfiles):
            fin = h5py.File(filename, 'r')
            try:
                for attr in ['Iterations', 'dt', 'nrx']:
                    _compare_attr(fout, fin, attr, filename)

                nrx = fin.attrs['nrx']
                for rx in range(1, nrx + 1):
                    path = '/rxs/rx' + str(rx)
                    if path not in fin:
                        raise CmdInputError('Receiver {} not found in {}'.format(rx, filename))
                    availableoutputs = list(fin[path].keys())
                    if availableoutputs != outputnames[rx]:
                        raise CmdInputError('Receiver {} outputs in {} do not match the first trace'.format(rx, filename))
                    if rx in positiondatasets:
                        positiondatasets[rx][model, :] = fin[path].attrs['Position']
                    for output in availableoutputs:
                        fout[path + '/' + output][:, model] = fin[path + '/' + output][:]
            finally:
                fin.close()
    finally:
        fout.close()

    if removefiles:
        for modelnumber, filename in outputfiles:
            os.remove(filename)

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Merges traces (A-scans) from multiple output files into one new file, then optionally removes the series of output files.', usage='cd gprMax; python -m tools.outputfiles_merge basefilename')
    parser.add_argument('basefilename', help='base name of output file series including path')
    parser.add_argument('--remove-files', action='store_true', default=False, help='flag to remove individual output files after merge')
    args = parser.parse_args()

    merge_files(args.basefilename, removefiles=args.remove_files)
