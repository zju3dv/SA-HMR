import torch
import torch.nn.functional as F
import numpy as np
from lib.utils.geo_transform import apply_T_on_points
from .utils import compute_similarity_transform_batch, solve_T_cr2c_with_2d3d_pnp, intersectionAndUnionGPU


def to_np(x):
    return x.cpu().numpy()


def to_list(x):
    return x.cpu().numpy().tolist()


def L2_error(x, y):
    return (x - y).pow(2).sum(-1).sqrt()


def MPVE_metric(evaluator, batch):
    gt_verts_3d = batch["gt_cr_verts"]  # (B, V, 3)
    pd_verts_3d = batch["pred_cr_verts"]  # (B, V, 3)
    l2 = L2_error(gt_verts_3d, pd_verts_3d) * 1000
    mean = to_list(l2.mean(-1))
    evaluator.update("MPVE", mean)
    if "pred_cr_verts_refined" in batch:
        l2 = L2_error(gt_verts_3d, batch["pred_cr_verts_refined"]) * 1000
        mean = to_list(l2.mean(-1))
        evaluator.update("MPVE_refined", mean)


def MPJPE_metric(evaluator, batch):
    gt_joints_3d = batch["gt_cr_joints14"]  # (B, J, 3)
    pd_joints_3d = batch["pred_cr_joints14"]  # (B, J, 3)
    l2 = L2_error(gt_joints_3d, pd_joints_3d) * 1000
    mean = to_list(l2.mean(-1))
    evaluator.update("MPJPE", mean)
    if "pred_cr_joints14_refined" in batch:
        l2 = L2_error(gt_joints_3d, batch["pred_cr_joints14_refined"]) * 1000
        mean = to_list(l2.mean(-1))
        evaluator.update("MPJPE_refined", mean)


def G_MPJPE_metric(evaluator, batch):
    if batch.get("flag_pred_coord", "cr") in ["c", "crm"]:
        gt_joints_3d = batch["gt_c_joints14"]  # (B, J, 3)
        pd_joints_3d = batch["pred_c_joints14"]
    else:
        gt_joints_3d = batch["gt_c_joints14"]  # (B, J, 3)
        if batch.get("pred_c_joints14", None) is not None:
            pd_joints_3d = batch["pred_c_joints14"]
        else:
            T_cr2c = solve_T_cr2c_with_2d3d_pnp(batch)
            pd_joints_3d = apply_T_on_points(batch["pred_cr_joints14"], T_cr2c)

            batch["pred_c_joints14"] = pd_joints_3d
            batch["pred_c_verts"] = apply_T_on_points(batch["pred_cr_verts"], T_cr2c)

        # visualize
        # from lib.utils.vis3d_utils import make_vis3d
        # vis3d = make_vis3d(None, 'global')
        # vis3d.add_point_cloud(gt_joints_3d[0], name='gt_joints3d')
        # vis3d.add_point_cloud(pd_joints_3d[0], name='pd_joints3d')

    l2 = L2_error(gt_joints_3d, pd_joints_3d) * 1000
    mean = to_list(l2.mean(-1))
    evaluator.update("G_MPJPE", mean)
    if "pred_c_joints14_refined" in batch:
        pd_joints_3d = batch["pred_c_joints14_refined"]
        l2 = L2_error(gt_joints_3d, pd_joints_3d) * 1000
        mean = to_list(l2.mean(-1))
        evaluator.update("G_MPJPE_refined", mean)


def G_MPVE_metric(evaluator, batch):
    gt_joints_3d = batch["gt_c_verts"]  # (B, J, 3)
    pd_joints_3d = batch["pred_c_verts"]  # (B, J, 3)
    l2 = L2_error(gt_joints_3d, pd_joints_3d) * 1000
    mean = to_list(l2.mean(-1))
    evaluator.update("G_MPVE", mean)


def PA_MPJPE_metric(evaluator, batch):
    gt_joints_3d = batch["gt_cr_joints14"]  # (B, J, 3)
    pd_joints_3d = batch["pred_cr_joints14"]  # (B, J, 3)
    pd_hat = compute_similarity_transform_batch(to_np(pd_joints_3d), to_np(gt_joints_3d))
    pd_hat = torch.from_numpy(pd_hat).to(pd_joints_3d.device)
    l2 = L2_error(pd_hat, gt_joints_3d) * 1000
    mean = to_list(l2.mean(-1))
    evaluator.update("PA_MPJPE", mean)
    if "pred_cr_joints14_refined" in batch:
        pd_joints_3d = batch["pred_cr_joints14_refined"]  # (B, J, 3)
        pd_hat = compute_similarity_transform_batch(to_np(pd_joints_3d), to_np(gt_joints_3d))
        pd_hat = torch.from_numpy(pd_hat).to(pd_joints_3d.device)
        l2 = L2_error(pd_hat, gt_joints_3d) * 1000
        mean = to_list(l2.mean(-1))
        evaluator.update("PA_MPJPE_refined", mean)


def C_HSCP_L2_metric(evaluator, batch):
    gt_points_3d = batch["gt_cr_verts"]  # (B, 6890, 3)
    pd_joints_3d = batch["pred_cr_verts"]  # (B, 6890, 3)
    l2 = L2_error(gt_points_3d, pd_joints_3d) * 1000
    contact = batch["gt_hsc"][..., 0]
    mean = []
    for b in range(contact.size(0)):
        mean.append((l2[b] * contact[b]).mean().cpu().numpy())
    evaluator.update("C_HSCP_L2", mean)

    if "pred_cr_verts_refined" in batch:
        gt_points_3d = batch["gt_cr_verts"]  # (B, 6890, 3)
        pd_joints_3d = batch["pred_cr_verts_refined"]  # (B, 6890, 3)
        l2 = L2_error(gt_points_3d, pd_joints_3d) * 1000
        contact = batch["gt_hsc"][..., 0]
        mean = []
        for b in range(contact.size(0)):
            mean.append((l2[b] * contact[b]).mean().cpu().numpy())
        evaluator.update("C_HSCP_L2_refined", mean)


@torch.no_grad()
def PelvisE_2D_metric(evaluator, batch):
    L2 = torch.norm(batch["gt_bi4s_pelvis_2d"].float() - batch["pred_bi4s_pelvis_2d"].float(), p=2, dim=-1)
    evaluator.update("PelvisE_2D", to_list(L2))


@torch.no_grad()
def PelvisE_metric(evaluator, batch):
    L2 = torch.norm(batch["gt_c_pelvis"] - batch["pred_c_pelvis"], 2, -1).squeeze(1) * 1000
    evaluator.update("PelvisE", to_list(L2))
    if "pred_c_pelvis_refined" in batch:
        L2 = torch.norm(batch["gt_c_pelvis"] - batch["pred_c_pelvis_refined"], 2, -1).squeeze(1) * 1000
        evaluator.update("PelvisE_refined", to_list(L2))

    if "pred_c_verts" in batch:
        pred_c_pelvis_mesh = batch["smpl"].get_h36m_pelvis(batch["pred_c_verts"])
        L2 = torch.norm(batch["gt_c_pelvis"] - pred_c_pelvis_mesh, 2, -1).squeeze(1) * 1000
        evaluator.update("PelvisE_refined_mesh", to_list(L2))


@torch.no_grad()
def HmSeg_metric(evaluator, batch):
    pred_hm = batch["pred_bi4s_seg_hm"]
    pred = pred_hm > 0.5
    gt = batch["gt_bi4s_seg_hm"]

    failed_recall, failed_pred = [], []
    for b in range(batch["B"]):
        failed_recall_, failed_pred_ = [], []
        for pid in range(7):
            pred_ = pred[b, pid].long()
            gt_ = gt[b, pid].long()
            intersection, union, target, estimate = intersectionAndUnionGPU(pred_, gt_, 2)

            if target[1] > 0:
                failed_recall_.append(1 - (intersection[1] / target[1]).cpu().numpy())
            if estimate[1] > 0:
                failed_pred_.append(1 - (intersection[1] / estimate[1]).cpu().numpy())

            # from lib.utils.vis3d_utils import make_vis3d
            # vis3d = make_vis3d(None, 'examine_prediction', time_postfix=True)
            # vis3d.add_image((batch['image'][0].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8), name='img')
            # vis3d.add_image((pred_ * 255).cpu().numpy().astype(np.uint8), name='pred')
            # vis3d.add_image((gt_ * 255).cpu().numpy().astype(np.uint8), name='gt')

        failed_recall.append(np.mean(failed_recall_) if len(failed_recall_) > 0 else 0)
        failed_pred.append(np.mean(failed_pred_) if len(failed_pred_) > 0 else 0)

    evaluator.update("HmSeg_RecallE", failed_recall)
    evaluator.update("HmSeg_PredE", failed_pred)


def VoxSeg_metric(evaluator, batch):
    pred = batch["pred_vox_segid"]
    gt = batch["gt_vox_seg_label"]
    prec, iou, recall = [], [], []
    for b in range(len(pred)):
        intersection, union, target, estimate = intersectionAndUnionGPU(pred[b], gt[b], 8)
        iou.append((sum(intersection[1:]) / (sum(union[1:]) + 1e-10)).cpu().numpy())
        recall.append((sum(intersection[1:]) / (sum(target[1:]) + 1e-10)).cpu().numpy())
        prec.append((sum(intersection[1:]) / (sum(estimate[1:]) + 1e-10)).cpu().numpy())
    evaluator.update("VoxSeg_PREC", prec)
    evaluator.update("VoxSeg_IOU", iou)
    evaluator.update("VoxSeg_RECALL", recall)


name2func = {
    # RCNet
    "VoxSeg": VoxSeg_metric,
    "PelvisE_2D": PelvisE_2D_metric,
    "PelvisE": PelvisE_metric,
    "HmSeg": HmSeg_metric,
    # SAHMR
    "MPJPE": MPJPE_metric,
    "MPVE": MPVE_metric,
    "PA_MPJPE": PA_MPJPE_metric,  # 评价姿势的相似程度（经过PA矫正）
    "G_MPJPE": G_MPJPE_metric,  # 评价世界坐标系下的姿态恢复程度
    "G_MPVE": G_MPVE_metric,
    "C_HSCP_L2": C_HSCP_L2_metric,  # 评价接触点的还原情况
}
