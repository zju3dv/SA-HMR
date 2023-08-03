import torch
import torch.nn.functional as F


def vertex_loss(trainer, batch, loss_stats):
    l1_loss = (batch["gt_cr_verts"] - batch["pred_cr_verts"]).abs().sum(-1)
    l1_loss = l1_loss.mean(-1)
    loss_stats["vertex_L1"] = l1_loss.detach().cpu()
    return l1_loss


def joint_loss(trainer, batch, loss_stats):
    l1_loss = (batch["gt_cr_joints14"] - batch["pred_cr_joints14"]).abs().sum(-1)
    l1_loss = l1_loss.mean(-1)
    loss_stats["joint_L1"] = l1_loss.detach().cpu()
    return l1_loss


def corr_points_loss(trainer, batch, loss_stats):
    pred_coord = batch[f"pred_crm_pcSeg_after"]
    gt_corr_pc = batch[f"pred_crm_pcSeg"]
    corr_mask = batch["pcSeg_mask"]
    l1_loss = (pred_coord - gt_corr_pc).abs().sum(-1)
    l1_loss = (l1_loss * corr_mask).mean(-1)
    loss_stats["corr_points_pred"] = l1_loss.detach().cpu()
    return l1_loss


def vertex_c_loss(trainer, batch, loss_stats):
    l1_loss = (batch["gt_c_verts"] - batch["pred_c_verts"]).abs().sum(-1)
    l1_loss = l1_loss.mean(-1)
    loss_stats["vertex_c_L1"] = l1_loss.detach().cpu()
    return l1_loss


def pelvis_hm_loss(trainer, batch, loss_stats):
    loss = (batch["gt_bi4s_pelvis_hm"] - batch["pred_bi4s_pelvis_hm"]).pow(2).sum((-1, -2))
    loss = loss.squeeze(1)
    loss_stats["pelvis_heatmap_MSE"] = loss.detach().cpu()
    return loss


def pelvis_depth_loss(trainer, batch, loss_stats):
    pred_bi4s_rdepth_hm = batch["pred_bi4s_rdepth_hm"]
    gt_bi4s_pelvis_2d = batch["gt_bi4s_pelvis_2d"]
    gkernel_3x3 = trainer.gkernel_3x3

    loss = []
    for b in range(pred_bi4s_rdepth_hm.size(0)):
        x, y = gt_bi4s_pelvis_2d[b]
        gt = batch["gt_bi_pelvis_zw1f"][b]
        pred = pred_bi4s_rdepth_hm[b, 0, y - 1 : y + 2, x - 1 : x + 2]
        loss.append(((gt - pred).abs() * gkernel_3x3).sum())
    loss = torch.stack(loss)
    loss_stats["pelvis_depth_L1"] = loss.detach().cpu()
    return loss


def bi4s_seg_hm_loss(trainer, batch, loss_stats):
    gt = batch["gt_bi4s_seg_hm"]
    pred = batch["pred_bi4s_seg_hm"]
    B, C, H, W = gt.shape
    # # Focal
    # score = pred.sigmoid()
    # loss = []
    # alpha, gamma = 0.01, 2
    # for b in range(B):
    #     # pos_loss = -alpha * score[b, gt[b] == 1].log() * (1-score[b, gt[b] == 1])**gamma
    #     # neg_loss = -(1-alpha) * (1-score[b, gt[b] == 0]).log() * (score[b, gt[b] == 0])**gamma
    #     pos_loss = - score[b, gt[b] == 1].log() * (1-score[b, gt[b] == 1])**gamma
    #     neg_loss = - (1-score[b, gt[b] == 0]).log() * (score[b, gt[b] == 0])**gamma
    #     loss.append(pos_loss.sum() + neg_loss.mean())
    # loss = torch.stack(loss)
    # loss_stats['bi4s_seg_hm_FL'] = loss.detach().cpu()

    # MSE
    loss = (gt - pred).pow(2).sum((-1, -2))
    loss = loss.sum(-1)
    loss_stats["bi4s_seg_hm_MSE"] = loss.detach().cpu()
    return loss


def bi4s_seg_hm_flip_loss(trainer, batch, loss_stats):
    if not trainer.training:
        return torch.zeros_like(loss_stats["bi4s_seg_hm_MSE"]).cuda()

    pred = batch["pred_bi4s_seg_hm_flip"]
    gt = batch["gt_bi4s_seg_hm"].flip([-1])
    # need extra flip on label
    gt = gt[:, [1, 0, 3, 2, 4, 5, 6]]

    # MSE
    loss = (gt - pred).pow(2).sum((-1, -2))
    loss = loss.sum(-1)
    loss_stats["bi4s_seg_hm_flip_MSE"] = loss.detach().cpu()
    return loss


def pcSeg_voxel_loss(trainer, batch, loss_stats):
    zero_scalar = batch["image"].new([0.0])
    label = batch["gt_vox_seg_label"]
    pred = batch["pred_vox_seg"]

    loss = []
    weight = torch.tensor([1.0] * 8).to(batch["image"])
    for b in range(len(pred)):
        if len(label[b]) > 0:
            loss.append(F.cross_entropy(pred[b], label[b], weight=weight, reduction="mean"))
        else:
            loss.append(zero_scalar)
    loss = torch.stack(loss)
    loss_stats["pcSeg_voxel"] = loss.detach().cpu()
    return loss


def vox_xyz_offset_loss(trainer, batch, loss_stats):
    zero_scalar = batch["image"].new([0.0])
    pred_vox_xyz_offset = batch["pred_vox_xyz_offset"]
    gt_vox_xyz_offset = batch["gt_vox_xyz_offset"]

    loss = []
    for b in range(len(pred_vox_xyz_offset)):
        if len(pred_vox_xyz_offset[b]) > 0:
            loss.append((pred_vox_xyz_offset[b] - gt_vox_xyz_offset[b]).abs().sum(-1).mean())
        else:
            loss.append(zero_scalar)
    loss = torch.stack(loss)
    loss_stats["vox_xyz_offset_L1"] = loss.detach().cpu()
    return loss


def pelvis_xyz_refined_loss(trainer, batch, loss_stats):
    pred_c_pelvis_refined = batch["pred_c_pelvis_refined"]
    gt_c_pelvis = batch["gt_c_pelvis"]

    loss = (pred_c_pelvis_refined - gt_c_pelvis).abs().sum(-1).squeeze(1)
    loss_stats["pelvis_xyz_refined_L1"] = loss.detach().cpu()
    return loss


name2loss = {
    # RCNet
    "pelvis_hm": pelvis_hm_loss,
    "pelvis_depth": pelvis_depth_loss,
    "pcSeg_voxel": pcSeg_voxel_loss,
    "vox_xyz_offset": vox_xyz_offset_loss,
    "pelvis_xyz_refined": pelvis_xyz_refined_loss,
    "bi4s_seg_hm": bi4s_seg_hm_loss,
    "bi4s_seg_hm_flip": bi4s_seg_hm_flip_loss,
    # SAHMR
    "vertex": vertex_loss,
    "joint": joint_loss,
    "vertex_c": vertex_c_loss,
    "corr_points": corr_points_loss,
}


# ----- Template ----- #


def loss_template(trainer, batch, loss_stats):
    """
    Args:
        trainer: to utilize the class variables
        batch, dict
        loss_stats, dict: record loss.detach().cpu()
    Returns:
        loss, torch.Tensor: (B,)
    """
    ...
