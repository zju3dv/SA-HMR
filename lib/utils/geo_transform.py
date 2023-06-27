import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pytorch3d.transforms import so3_exp_map, so3_log_map
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle
import pytorch3d.ops.knn as knn


def homo_points(points):
    """
    points: (B, N, 3)
    Returns: (B, N, 4), with 1 padded
    """
    return F.pad(points, [0, 1], value=1.0)


def apply_T_on_points(points, T):
    """
    points: (B, N, 3)
    T: (B, 4, 4)
    Returns: (B, N, 3)
    """
    points_h = homo_points(points)
    points_T_h = torch.einsum("bki,bji->bjk", T, points_h)
    return points_T_h[..., :3].contiguous()


def project_p2d(points, K):
    """
    points: (B, N, 3)
    K: (B, 3, 3)
    Returns: (B, N, 2)
    """
    points_proj = points / points[:, :, [-1]]
    p2d_h = torch.einsum("bki,bji->bjk", K, points_proj)
    return p2d_h[..., :2]


def gen_uv_from_HW(H, W, device="cpu"):
    """Returns: (H, W, 2), as float. Note: uv not ij"""
    grid_v, grid_u = torch.meshgrid(torch.arange(H), torch.arange(W))
    return (
        torch.stack(
            [grid_u, grid_v],
            dim=-1,
        )
        .float()
        .to(device)
    )  # (H, W, 2)


def unproject_p2d(uv, z, K):
    """we assume a pinhole camera for unprojection
    uv: (B, N, 2)
    z: (B, N, 1)
    K: (B, 3, 3)
    Returns: (B, N, 3)
    """
    xy_atz1 = (uv - K[:, None, :2, 2]) / K[:, None, [0, 1], [0, 1]]  # (B, N, 2)
    xyz = torch.cat([xy_atz1 * z, z], dim=-1)
    return xyz


def cvt_to_bi01_p2d(p2d, bbx_lurb):
    """
    p2d: (B, N, 2)
    bbx_lurb: (B, 4)
    """
    bbx_wh = bbx_lurb[:, None, 2:] - bbx_lurb[:, None, :2]
    bi01_p2d = (p2d - bbx_lurb[:, None, :2]) / bbx_wh
    return bi01_p2d


def cvt_from_bi01_p2d(bi01_p2d, bbx_lurb):
    """
    p2d: (B, N, 2)
    bbx_lurb: (B, 4)
    """
    bbx_wh = bbx_lurb[:, None, 2:] - bbx_lurb[:, None, :2]
    p2d = (bi01_p2d * bbx_wh) + bbx_lurb[:, None, :2]
    return p2d


def uv2l_index(uv, W):
    return uv[..., 0] + uv[..., 1] * W


def l2uv_index(l, W):
    v = torch.div(l, W, rounding_mode="floor")
    u = l % W
    return torch.stack([u, v], dim=-1)


def transform_mat(R, t):
    """
    Args:
        R: Bx3x3 array of a batch of rotation matrices
        t: Bx3x(1) array of a batch of translation vectors
    Returns:
        T: Bx4x4 Transformation matrix
    """
    # No padding left or right, only add an extra row
    if len(R.shape) > len(t.shape):
        t = t[..., None]
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=-1)


def axis_angle_to_matrix_exp_map(aa):
    """use pytorch3d so3_exp_map
    Args:
        aa: (*, 3)
    Returns:
        R: (*, 3, 3)
    """
    print("Use pytorch3d.transforms.axis_angle_to_matrix instead!!!")
    ori_shape = aa.shape[:-1]
    return so3_exp_map(aa.reshape(-1, 3)).reshape(*ori_shape, 3, 3)


def matrix_to_axis_angle_log_map(R):
    """use pytorch3d so3_log_map
    Args:
        aa: (*, 3, 3)
    Returns:
        R: (*, 3)
    """
    print("WARINING! I met singularity problem with this function, use matrix_to_axis_angle instead!")
    ori_shape = R.shape[:-2]
    return so3_log_map(R.reshape(-1, 3, 3)).reshape(*ori_shape, 3)


def matrix_to_axis_angle(R):
    """use pytorch3d so3_log_map
    Args:
        aa: (*, 3, 3)
    Returns:
        R: (*, 3)
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(R))


def ransac_PnP(K, pts_2d, pts_3d, err_thr=10):
    """solve pnp"""
    dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")

    pts_2d = np.ascontiguousarray(pts_2d.astype(np.float64))
    pts_3d = np.ascontiguousarray(pts_3d.astype(np.float64))
    K = K.astype(np.float64)

    try:
        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, K, dist_coeffs, reprojectionError=err_thr, iterationsCount=10000, flags=cv2.SOLVEPNP_EPNP
        )

        rotation = cv2.Rodrigues(rvec)[0]

        pose = np.concatenate([rotation, tvec], axis=-1)
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)

        inliers = [] if inliers is None else inliers

        return pose, pose_homo, inliers
    except cv2.error:
        print("CV ERROR")
        return np.eye(4)[:3], np.eye(4), []


def ransac_PnP_batch(K_raw, pts_2d, pts_3d, err_thr=10):
    fit_R, fit_t = [], []
    for b in range(K_raw.shape[0]):
        pose, _, inliers = ransac_PnP(K_raw[b], pts_2d[b], pts_3d[b], err_thr=err_thr)
        fit_R.append(pose[:3, :3])
        fit_t.append(pose[:3, 3])
    fit_R = np.stack(fit_R, axis=0)
    fit_t = np.stack(fit_t, axis=0)
    return fit_R, fit_t


def get_nearby_points(points, query_verts, padding=0.0, p=1):
    """
    points: (S, 3)
    query_verts: (V, 3)
    """
    if p == 1:
        max_xyz = query_verts.max(0)[0] + padding
        min_xyz = query_verts.min(0)[0] - padding
        idx = (((points - min_xyz) > 0).all(dim=-1) * ((points - max_xyz) < 0).all(dim=-1)).nonzero().squeeze(-1)
        nearby_points = points[idx]
    elif p == 2:
        squared_dist, _, _ = knn.knn_points(points[None], query_verts[None], K=1, return_nn=False)
        mask = squared_dist[0, :, 0] < padding**2  # (S,)
        nearby_points = points[mask]

    return nearby_points


def unproj_bbx_to_fst(bbx_lurb, K, near_z=0.5, far_z=12.5):
    B = bbx_lurb.size(0)
    uv = bbx_lurb[:, [[0, 1], [2, 1], [2, 3], [0, 3], [0, 1], [2, 1], [2, 3], [0, 3]]]
    if isinstance(near_z, float):
        z = uv.new([near_z] * 4 + [far_z] * 4).reshape(1, 8, 1).repeat(B, 1, 1)
    else:
        z = torch.cat([near_z[:, None, None].repeat(1, 4, 1), far_z[:, None, None].repeat(1, 4, 1)], dim=1)
    c_frustum_points = unproject_p2d(uv, z, K)  # (B, 8, 3)
    return c_frustum_points
