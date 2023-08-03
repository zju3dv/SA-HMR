import torch
import numpy as np
from lib.utils.geo_transform import ransac_PnP_batch, transform_mat


def to_np(x):
    return x.cpu().numpy()


# ===== PA-MPJPE ===== #


def compute_similarity_transform(S1, S2):
    """Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat


# ===== G-MPJPE ===== #


def solve_T_cr2c_with_2d3d_pnp(batch):
    HW = batch["image"].shape[-1]  # 224
    bbx_lurb = batch["bbx_lurb"][:, None]  # (B, 1, 4)
    pred_joints14 = batch["pred_cr_joints14"]  # (B, J, 3)

    # 1. 将投影得到的关键点，计算到原图中
    pred_ri_joints14_2d = (
        batch["pred_bi_joints14_2d"] * ((bbx_lurb[..., 2:] - bbx_lurb[..., :2]) / (HW - 1)) + bbx_lurb[..., :2]
    )
    # 2. 用2D-3D correspondence计算得到 T_m2w
    fit_R, fit_t = ransac_PnP_batch(to_np(batch["K"]), to_np(pred_ri_joints14_2d), to_np(pred_joints14), err_thr=10)
    T_cr2c = transform_mat(torch.FloatTensor(fit_R), torch.FloatTensor(fit_t)).to(batch["T_w2c"].device)
    return T_cr2c


# ===== Seg IoU ===== #


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target, area_output
