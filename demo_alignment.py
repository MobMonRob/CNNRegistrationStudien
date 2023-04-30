import os
import math
import numpy as np
import argparse
import open3d as o3d
import MinkowskiEngine as ME
import torch
import typing as t
import util.transform_estimation as te

from urllib.request import urlretrieve
from model.resunet import ResUNetBN2C
from lib.eval import find_nn_gpu

if not os.path.isfile('ResUNetBN2C-16feat-3conv.pth'):
    print('Downloading weights...')
    urlretrieve(
        "https://node1.chrischoy.org/data/publications/fcgf/2019-09-18_14-15-59.pth",
        'ResUNetBN2C-16feat-3conv.pth')

NN_MAX_N = 2500
SUBSAMPLE_SIZE = 15000


def points_to_pointcloud(
        points: np.array, voxel_size: float = 0.025, scalars: t.Optional[np.array] = None
) -> o3d.geometry.PointCloud():
    """ convert numpy array points to open3d.PointCloud
    :param points: np.ndarray of shape (N, 3) representing floating point coordinates
    :param voxel_size: float
    :param scalars: (optional) np.ndarray of shape (N, 1), scalar of each point (e.g. FDI)
    :return: open3d.PointCloud
    """
    radius_normal = voxel_size * 2
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    pcd.estimate_covariances(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    if scalars is not None:
        colors = np.asarray([int_to_rgb(i) for i in scalars])
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def int_to_rgb(val: int, min_val: int = 11, max_val: int = 48, norm: bool = True):
    if val > max_val:
        raise ValueError("val must not be greater than max_val")
    if val < 0 or max_val < 0:
        raise ValueError("arguments may not be negative")
    if val < min_val:
        raise ValueError("val must be greater than min_val")

    i = (val - min_val) * 255 / (max_val - min_val)
    r = round(math.sin(0.024 * i + 0) * 127 + 128)
    g = round(math.sin(0.024 * i + 2) * 127 + 128)
    b = round(math.sin(0.024 * i + 4) * 127 + 128)
    if norm:
        r /= 255
        g /= 255
        b /= 255
    return [r, g, b]

def apply_transformation(
        points: t.Union[np.ndarray, torch.Tensor],
        transformation: t.Union[np.ndarray, torch.Tensor],
        padding: float = 0.3, # add 30 cm padding in all directions
) -> t.Union[np.ndarray, torch.Tensor]:
    """
    :param points: tensor of shape (N, 3) representing floating point coordinates
    :param transformation: (4, 4) tensor of a transformation matrix
    :return: transformed points
    """

    # add padding to the input points
    points[:, 0] += padding
    points[:, 1] += padding
    points[:, 2] += padding

    if all(isinstance(i, np.ndarray) for i in [points, transformation]):
        transformed_points = np.matmul(
            transformation,
            np.concatenate(
                [points[:, 0:3], np.ones(shape=(points.shape[0], 1))], axis=-1
            ).T,
        ).T
    elif all(isinstance(i, torch.Tensor) for i in [points, transformation]):
        transformed_points = torch.matmul(
            transformation,
            torch.concat(
                [points[:, 0:3], torch.ones(size=(points.shape[0], 1))], dim=-1
            ).T,
        ).T
    else:
        raise TypeError("Both inputs should be either np.ndarray or torch.Tensor type.")
    points[:, 0:3] = transformed_points[:, 0:3]
    return points


def find_corr(xyz0, xyz1, F0, F1, subsample_size=-1):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
        # amount of points in F0 and F1
        N0 = min(len(F0), subsample_size)
        N1 = min(len(F1), subsample_size)
        inds0 = np.random.choice(len(F0), N0, replace=False)
        inds1 = np.random.choice(len(F1), N1, replace=False)
        F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn, until all neighbours are found
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=NN_MAX_N)
    if subsample_size > 0 and subsample:
        return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
        return xyz0, xyz1[nn_inds]


def demo_alignment(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    voxel_size = config.voxel_size
    checkpoint = torch.load(config.model)

    # init model
    model = ResUNetBN2C(1, 16, normalize_feature=True, conv1_kernel_size=3, D=3)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.to(device)


    # create input1 input (points and features)
    input1_pcd = o3d.io.read_point_cloud(config.input1)
    input1_xyz = np.array(input1_pcd.points)
    input1_feats = np.ones((len(input1_xyz), 1))


    # create input2 input (points and features)
    input2_pcd = o3d.io.read_point_cloud(config.input2)
    input2_xyz = np.array(input2_pcd.points)
    input2_feats = np.ones((len(input2_xyz), 1))

    # create input1 sparse tensor and model features
    # voxelize xyz and feats
    input1_coords = np.floor(input1_xyz / voxel_size)
    input1_coords, input1_inds = ME.utils.sparse_quantize(input1_coords, return_index=True)
    # convert to batched coords compatible with ME
    input1_coords = ME.utils.batched_coordinates([input1_coords])
    input1_unique_xyz = input1_xyz[input1_inds]
    input1_feats = input1_feats[input1_inds]
    input1_tensor = ME.SparseTensor(
        torch.tensor(input1_feats, dtype=torch.float32),
        coordinates=torch.tensor(input1_coords, dtype=torch.int32),
        device=device
    )

    # create input2 sparse tensor and model features
    input2_coords = np.floor(input2_xyz / voxel_size)
    input2_coords, input2_inds = ME.utils.sparse_quantize(input2_coords, return_index=True)
    # convert to batched coords compatible with ME
    input2_coords = ME.utils.batched_coordinates([input2_coords])
    input2_unique_xyz = input2_xyz[input2_inds]
    input2_feats = input2_feats[input2_inds]
    input2_tensor = ME.SparseTensor(
        torch.tensor(input2_feats, dtype=torch.float32),
        coordinates=torch.tensor(input2_coords, dtype=torch.int32),
        device=device
    )

    # save the aligned point clouds to PLY files
    o3d.io.write_point_cloud('input1.ply', points_to_pointcloud(input1_unique_xyz))
    o3d.io.write_point_cloud('input2.ply', points_to_pointcloud(input2_unique_xyz))


    # get model features of inputs
    input1_model_feats = model(input1_tensor).F
    input2_model_feats = model(input2_tensor).F

    # compute correspondences and alignment
    xyz0_corr, xyz1_corr = find_corr(
        torch.tensor(input2_unique_xyz, dtype=torch.float32).to(device),
        torch.tensor(input1_unique_xyz, dtype=torch.float32).to(device),
        input2_model_feats,
        input1_model_feats,
        subsample_size=SUBSAMPLE_SIZE,
    )
    xyz0_corr, xyz1_corr = xyz0_corr.cpu(), xyz1_corr.cpu()

    # estimate transformation using the correspondences,  robuste quadratisch-lineare Transformation, it also uses ranasac for the transformation
    est_transformation = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

    aligned_input2 = apply_transformation(input2_xyz.copy(), est_transformation.numpy())

    # save the aligned point clouds to PLY files
    o3d.io.write_point_cloud('input1_aligned.ply', points_to_pointcloud(input1_xyz))
    o3d.io.write_point_cloud('input2_aligned.ply', points_to_pointcloud(aligned_input2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i1',
        '--input1',
        default='/path/to/file',
        type=str,
        help='/path/to/file')
    parser.add_argument(
        '-i2',
        '--input2',
        default='/path/to/file',
        type=str,
        help='/path/to/file')
    parser.add_argument(
        '-m',
        '--model',
        default='ResUNetBN2C-16feat-3conv.pth',
        type=str,
        help='path to latest checkpoint (default: None)')
    parser.add_argument(
        '--voxel_size',
        default=0.4, # needs to correspond to the voxel size used in downsampling
        type=float,
        help='voxel size to preprocess point cloud')

    config = parser.parse_args()
    demo_alignment(config)
