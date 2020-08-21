"""
Deeban Ramalingam (deebanr@slac.stanford.edu)

The goal of this script is to efficiently assemble the diffraction intensity volume using the MPI Communication Model.

Run instructions:

mpiexec -n 48 python assemble_diffraction_intensity_volume_mpi.py --config assemble-diffraction-intensity-volume-mpi.json --dataset 3iyf-10K

Algorithm for assembling the diffraction intensity volume using the MPI Communication Model:

1. Master creates an empty intensity grid
2. Slave i asks Master for data index k
3. Master provides Slave i with data index k
4. Slave i creates an empty intensity grid k
5. Slave i rotates intensity grid k using orientation k
6. Slave i adds diffraction pattern k to intensity grid k using nearest-neighbor interpolation
7. Slave i sends intensity grid k to Reducer
8. Reducer adds intensity grid k to the intensity grid created by Master
9. Repeat steps 2-8 for all k

Note that Master, Slaves, and Reducer all work in parallel.
"""

# MPI parameters
from mpi4py import MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
N_RANKS = COMM.size

if RANK == 0:
    assert N_RANKS >= 3, "This script is planned for at least 2 ranks."

MASTER_RANK = 0
REDUCER_RANK = N_RANKS - 1

import time
import os

# Unlock parallel but non-MPI HDF5
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

from os.path import dirname, abspath

import sys
import argparse
import json

import numpy as np

from tqdm import tqdm

import h5py as h5

def main():
    user_input = parse_input_arguments(sys.argv)
    config_file = user_input['config']
    dataset_name = user_input['dataset'] 

    with open(config_file) as config_file:
        config_params = json.load(config_file)
    
    if dataset_name not in config_params:
        raise Exception("Dataset {} not in Config file.".format(dataset_name))
    
    dataset_params = config_params[dataset_name]
    h5_file = dataset_params["h5File"]
    downsampled_h5_file = dataset_params["downsampledH5File"]
    diffraction_intensity_volume_h5_file = dataset_params["diffractionIntensityVolumeH5File"]
    dataset_size = dataset_params["datasetSize"]
    
    if RANK == MASTER_RANK:
        print("\n(Master) Create empty intensity grid.")
        intensity_coords, intensity_vals = build_empty_intensity_grid()
        
        diffraction_intensity_volume_h5_file_parent_directory = dirname(abspath(diffraction_intensity_volume_h5_file))
        if not os.path.exists(diffraction_intensity_volume_h5_file_parent_directory):
            print("\n(Master) Create directory: {}.".format(diffraction_intensity_volume_h5_file_parent_directory))
            os.makedirs(diffraction_intensity_volume_h5_file_parent_directory)
        
        
        print("\n(Master) Create H5 file to save diffraction intensity volume: {}.".format(diffraction_intensity_volume_h5_file))
        diffraction_intensity_volume_h5_file_handle = h5.File(diffraction_intensity_volume_h5_file, 'w')
        
        diffraction_intensity_volume_h5_file_handle.create_dataset("intensity_coords", dtype='f', data=intensity_coords)
        diffraction_intensity_volume_h5_file_handle.create_dataset("intensity_vals", dtype='f', data=intensity_vals)
        
        diffraction_intensity_volume_h5_file_handle.close()
    
    sys.stdout.flush()
    COMM.barrier()
    
    if RANK == MASTER_RANK:
        
        print("\n(Master) Start receiving requests for data from Slaves.")
        
        for data_k in range(dataset_size):
            
            slave_i = COMM.recv(source=MPI.ANY_SOURCE)
            COMM.send(data_k, dest=slave_i)
        
        n_slaves = N_RANKS - 2
        for _ in range(n_slaves):
            
            # Send one "None" to each rank as final flag to stop asking for more data
            slave_i = COMM.recv(source=MPI.ANY_SOURCE)
            COMM.send(None, dest=slave_i)
    
    elif RANK == REDUCER_RANK:
        
        diffraction_intensity_volume_h5_file_handle = h5.File(diffraction_intensity_volume_h5_file, 'r+')

        intensity_vals = diffraction_intensity_volume_h5_file_handle["intensity_vals"][:]
                
        n_diffraction_patterns_processed = 0
        progress_bar_n_diffraction_patterns_processed = tqdm(total=dataset_size)
        
        while n_diffraction_patterns_processed < dataset_size:
            
            intensity_vals_to_add = COMM.recv(source=MPI.ANY_SOURCE)
            intensity_vals += intensity_vals_to_add
            
            n_diffraction_patterns_processed += 1
            progress_bar_n_diffraction_patterns_processed.update(1)
        
        diffraction_intensity_volume_h5_file_handle["intensity_vals"][:] = intensity_vals
        
        diffraction_intensity_volume_h5_file_handle.close()
        
    else:

        h5_file_handle = h5.File(h5_file, 'r')
        downsampled_h5_file_handle = h5.File(downsampled_h5_file, 'r')
        diffraction_intensity_volume_h5_file_handle = h5.File(diffraction_intensity_volume_h5_file, 'r')
        
        intensity_coords = diffraction_intensity_volume_h5_file_handle["intensity_coords"]
        intensity_vals = diffraction_intensity_volume_h5_file_handle["intensity_vals"][:]
        
        while True:
            
            COMM.send(RANK, dest=MASTER_RANK)
            data_k = COMM.recv(source=MASTER_RANK)
            
            if data_k is None:
                print("\n(Slave {}) Receive final flag from Master to stop asking for more data.".format(RANK))
                break
            
            diffraction_pattern = downsampled_h5_file_handle["downsampled_diffraction_patterns"][data_k]
            
            orientation = h5_file_handle["orientations"][data_k]
            rotation_matrix_3d = quat2mat(orientation)
            
            oriented_intensity_coords = np.dot(intensity_coords, rotation_matrix_3d)

            intensity_vals_to_add = np.zeros_like(intensity_vals)
            
            interpolate_oriented_intensity_using_diffraction_pattern(oriented_intensity_coords, intensity_vals_to_add, diffraction_pattern)
            
            COMM.send(intensity_vals_to_add, dest=REDUCER_RANK)
        
        h5_file_handle.close()
        downsampled_h5_file_handle.close()
        diffraction_intensity_volume_h5_file_handle.close()
    
    sys.stdout.flush()
    COMM.barrier()

# Adapted from: https://sscc.nimh.nih.gov/pub/dist/bin/linux_gcc32/meica.libs/nibabel/quaternions.py
def quat2mat(q):
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    FLOAT_EPS = np.finfo(np.float).eps
    if Nq < FLOAT_EPS:
        return np.eye(3)
    
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    
    return np.array(
           [[ 1.0 - (yY + zZ), xY - wZ, xZ + wY ],
            [ xY + wZ, 1.0 - (xX + zZ), yZ - wX ],
            [ xZ - wY, yZ + wX, 1.0 - (xX + yY) ]])

def build_empty_intensity_grid():
    x_ = np.linspace(-63., 64., 128.)
    y_ = np.linspace(-63., 64., 128.)
    z_ = np.linspace(-63., 64., 128.)

    x, y, z = np.meshgrid(x_, y_, z_)

    intensity_coords = np.column_stack((x.flatten(), y.flatten(), z.flatten()))

    intensity_vals = np.zeros(len(intensity_coords))

    return intensity_coords, intensity_vals

def interpolate_oriented_intensity_using_diffraction_pattern(oriented_intensity_coords, intensity_vals, diffraction_pattern):
    n_oriented_intensity_coords = len(oriented_intensity_coords)
    diffraction_pattern_height = diffraction_pattern.shape[0]
    diffraction_pattern_width = diffraction_pattern.shape[1]
    
    for oriented_intensity_coord_index in range(n_oriented_intensity_coords):
        
        oriented_intensity_coord_z = oriented_intensity_coords[oriented_intensity_coord_index, 2]

        diffraction_slice_coord_z = int(round(oriented_intensity_coord_z))
        
        if diffraction_slice_coord_z == 0:
            
            oriented_intensity_coord_x = oriented_intensity_coords[oriented_intensity_coord_index, 0]
            oriented_intensity_coord_y = oriented_intensity_coords[oriented_intensity_coord_index, 1]
        
            diffraction_slice_coord_x = int(round(oriented_intensity_coord_x))
            diffraction_slice_coord_y = int(round(oriented_intensity_coord_y))
            
            diffraction_pattern_x = diffraction_slice_coord_x + diffraction_pattern_height // 2 - 1
            diffraction_pattern_y = diffraction_slice_coord_y + diffraction_pattern_width // 2 - 1
            
            if 0 <= diffraction_pattern_x and diffraction_pattern_x < diffraction_pattern_height and 0 <= diffraction_pattern_y and diffraction_pattern_y < diffraction_pattern_width: 
                intensity_vals[oriented_intensity_coord_index] += diffraction_pattern[diffraction_pattern_x, diffraction_pattern_y]

def parse_input_arguments(args):
    del args[0]
    
    parse = argparse.ArgumentParser()
    parse.add_argument('-c', '--config', type=str, help='JSON Config file')
    parse.add_argument('-d', '--dataset', type=str, help='Dataset name: 3iyf-10K')

    return vars(parse.parse_args(args))

if __name__ == '__main__':
    main()
