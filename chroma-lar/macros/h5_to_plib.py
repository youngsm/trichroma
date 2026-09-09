#!/usr/bin/env python
import numpy as np
import h5py
from photonlib import PhotonLib, VoxelMeta
from utils.log import logger

def h5_to_plib(file_path: str, output_path: str):
    
    logger.info(f"reading {file_path}")
    with h5py.File(file_path, "r") as f:
        if 'numvox' in f:
            raise ValueError("This file is already in PhotonLib format.")

        x = f["posX"]
        y = f["posY"]
        z = f["posZ"]
        positions = np.array([x, y, z]).T
        pte = np.array(
            [f[k] for k in f.keys() if k.startswith("ch")]
        ).T
    logger.info(f"read {positions.shape[0]} positions and {pte.shape[1]} channels")
    
    # 1. find pitch per coordinate, i.e. smallest non-zero difference
    # between any two positions
    diffs = positions - positions[0]
    pitch = []
    for i in range(3):
        col = np.abs(diffs[:, i])
        pitch.append(np.min(np.abs(col[col != 0])))
    pitch = np.array(pitch)
    logger.info(f"found pitches: {pitch}")

    # 2. generate bbox bounds
    bbox_min = []
    bbox_max = []
    for i in range(3):
        bbox_min.append(np.min(positions[:, i]) - pitch[i] / 2)
        bbox_max.append(np.max(positions[:, i]) + pitch[i] / 2)
    bounds = np.array([bbox_min, bbox_max])
    logger.info(f"found bounds: {bounds.tolist()}")

    # 3. Create lightmap grid
    x = np.arange(bounds[0, 0] + pitch[0] / 2, bounds[1, 0], pitch[0])
    y = np.arange(bounds[0, 1] + pitch[1] / 2, bounds[1, 1], pitch[1])
    z = np.arange(bounds[0, 2] + pitch[2] / 2, bounds[1, 2], pitch[2])

    # create grid of all positions
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    all_pos = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    voxel_shape = (len(x), len(y), len(z))
    
    # we need to reverse the order of the positions to match the order that
    # PhotonLib expects. we'll also need to sort the pte values.
    sorted_indices = np.lexsort((all_pos[:, 0], all_pos[:, 1], all_pos[:, 2]))
    all_pos = all_pos[sorted_indices]

    # 4. Fill lightmap grid with values, setting zero for missing values
    voxel_shape = (len(x), len(y), len(z))
    full_values = np.full(shape=(all_pos.shape[0], pte.shape[1]), fill_value=0, dtype=float)
    starting_point = bbox_min + pitch / 2
    indices = np.round((positions - starting_point) / pitch).astype(int)
    linear_indices = np.ravel_multi_index(indices.T, voxel_shape)
    full_values[linear_indices] = pte
    full_values = full_values[sorted_indices]
    logger.info(f"lightmap {100*len(positions)/len(all_pos):.1f}% filled with PTE values")

    # 5. Create PhotonLib object
    meta = VoxelMeta(shape=voxel_shape, ranges=bounds.T)
    PhotonLib.save(output_path, full_values, meta)

def plot_photonlib(plib_path):
    import trimesh
    
    plib = PhotonLib.load(plib_path)
    vis = plib.vis
    positions = plib.meta.voxel_to_coord(range(vis.shape[0]))

    pc = trimesh.PointCloud(positions)
    pc.colors = trimesh.visual.color.interpolate(
        np.log10(vis.mean(axis=1) + 1e-8), "viridis"
    )
    scene = trimesh.Scene([pc])
    scene.show()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert photon library from HDF5 to PhotonLib format")
    parser.add_argument("positions", type=str, help="Path to photon library HDF5 file")
    parser.add_argument("output", type=str, help="Path to output PhotonLib file")
    parser.add_argument("--vis", action="store_true", help="Visualize the lightmap")    
    
    args = parser.parse_args()

    h5_to_plib(args.positions, args.output)
    
    if args.vis:
        plot_photonlib(args.output)
    
if __name__ == "__main__":
    main()