import torch
import torch.nn.functional as F


def vertex431_refined_loss(trainer, batch, loss_stats):
    l1_loss = (batch['gt_c_verts431'] - batch['pred_c_verts431_refined']).abs().sum(-1)
    l1_loss = l1_loss.mean(-1)
    loss_stats['verts431_c_refined_L1'] = l1_loss.detach().cpu()
    return l1_loss


def vertex_offset_loss(trainer, batch, loss_stats):
    zero_scalar = batch['image'].new([0.])
    pred = batch['pred_vox_verts431_offset']
    gt = batch['gt_vox_verts431_offset']  # [N', 431, 3]

    loss = []
    for b in range(batch['B']):
        if len(pred[b]) > 0:
            loss.append((pred[b] - gt[b]).abs().sum(-1).mean(-1).mean(-1))
        else:
            loss.append(zero_scalar)
    loss = torch.stack(loss)
    loss_stats['vertex_offset'] = loss.detach().cpu()
    return loss


def vertex_refined_loss(trainer, batch, loss_stats):
    loss = (batch['pred_c_verts_refined'] - batch['gt_c_verts']).abs().sum(-1).mean(-1)
    loss_stats['vertex_refined'] = loss.detach().cpu()
    return loss


def vertex_cr_refined_loss(trainer, batch, loss_stats):
    l1_loss = (batch['gt_cr_verts'] - batch['pred_cr_verts_refined']).abs().sum(-1)
    l1_loss = l1_loss.mean(-1)
    loss_stats['vertex_cr_refined_L1'] = l1_loss.detach().cpu()
    return l1_loss


def joint_cr_refined_loss(trainer, batch, loss_stats):
    l1_loss = (batch['gt_cr_joints14'] - batch['pred_cr_joints14_refined']).abs().sum(-1)
    l1_loss = l1_loss.mean(-1)
    loss_stats['joint_cr_refined_L1'] = l1_loss.detach().cpu()
    return l1_loss


def vertex_loss(trainer, batch, loss_stats):
    l1_loss = (batch['gt_cr_verts'] - batch['pred_cr_verts']).abs().sum(-1)
    l1_loss = l1_loss.mean(-1)
    loss_stats['vertex_L1'] = l1_loss.detach().cpu()
    return l1_loss


def vertex_c_loss(trainer, batch, loss_stats):
    l1_loss = (batch['gt_c_verts'] - batch['pred_c_verts']).abs().sum(-1)
    l1_loss = l1_loss.mean(-1)
    loss_stats['vertex_c_L1'] = l1_loss.detach().cpu()
    return l1_loss


def vertex_contact_l1_loss(trainer, batch, loss_stats):
    l1_loss = (batch['gt_cr_verts'] - batch['pred_cr_verts']).abs().sum(-1)  # (B, 6890)
    l1_loss = (l1_loss * batch['gt_hsc'][..., 0]).mean(-1)
    loss_stats['vertex_contact_L1'] = l1_loss.detach().cpu()
    return l1_loss


def vertex_contact_mse_loss(trainer, batch, loss_stats):
    l2_loss = (batch['gt_cr_verts'] - batch['pred_cr_verts']).pow(2).sum(-1)  # (B, 6890)
    l2_loss = (l2_loss * batch['gt_hsc'][..., 0]).mean(-1)
    loss_stats['vertex_contact_L2'] = l2_loss.detach().cpu()
    return l2_loss


def joint2d_loss(trainer, batch, loss_stats):
    """ this may help orthogonal cam-params """
    pred_joints14_2d = batch['pred_bi_joints14_2d'] / batch['image'].size(-1)
    gt_2d_joints14 = batch['gt_bi_joints14_2d'] / batch['image'].size(-1)
    l1_loss = (pred_joints14_2d - gt_2d_joints14).abs().sum(-1)
    l1_loss = l1_loss.mean(-1)
    loss_stats['joints_2d_L1'] = l1_loss.detach().cpu()
    return l1_loss


def joint_loss(trainer, batch, loss_stats):
    l1_loss = (batch['gt_cr_joints14'] - batch['pred_cr_joints14']).abs().sum(-1)
    l1_loss = l1_loss.mean(-1)
    loss_stats['joint_L1'] = l1_loss.detach().cpu()
    return l1_loss


def joint_c_loss(trainer, batch, loss_stats):
    l1_loss = (batch['gt_c_joints14'] - batch['pred_c_joints14']).abs().sum(-1)
    l1_loss = l1_loss.mean(-1)
    loss_stats['joint_L1'] = l1_loss.detach().cpu()
    return l1_loss


def corr_points_pred_loss(trainer, batch, loss_stats):
    pred_coord = batch.get('flag_pred_coord', 'cr')
    corr_pc_type = batch['flag_corr_pc_type']
    if corr_pc_type == 'points431':
        gt_corr_pc = batch[f'{pred_coord}_corr_points431']
        corr_mask = batch['corr_mask431']
    elif corr_pc_type in ['pcSeg', 'pcSeg_wocorr']:
        gt_corr_pc = batch[f'{pred_coord}_pcSeg']
        corr_mask = batch['pcSeg_mask']

    l1_loss = (batch[f'pred_{pred_coord}_corr_pc'] - gt_corr_pc).abs().sum(-1)
    l1_loss = (l1_loss * corr_mask).mean(-1)
    loss_stats['corr_points_pred'] = l1_loss.detach().cpu()
    return l1_loss


def corr_points_pred_immerse_loss(trainer, batch, loss_stats):
    pred_coord = batch[f'pred_crm_pcSeg_after']
    gt_corr_pc = batch[f'pred_crm_pcSeg']
    corr_mask = batch['pcSeg_mask']
    l1_loss = (pred_coord - gt_corr_pc).abs().sum(-1)
    l1_loss = (l1_loss * corr_mask).mean(-1)
    loss_stats['corr_points_pred'] = l1_loss.detach().cpu()
    return l1_loss


def feat_sim_cpv_loss(trainer, batch, loss_stats):
    raise NotImplementedError
    loss = 0
    for i in range(1, 3):
        feat1 = batch[f'cpose_img_feat_{i}'][:, 14:]
        feat2 = batch[f'cpose_cp_feat_{i}'].detach()
        loss += ((1 - F.cosine_similarity(feat1, feat2, dim=2)) * batch['corr_mask431']).mean()
    loss_stats['feat_sim_cpv'] = loss.detach().cpu()
    return loss


def corr_loss(trainer, batch, loss_stats):
    ...


def pelvis_hm_loss(trainer, batch, loss_stats):
    loss = (batch['gt_bi4s_pelvis_hm'] - batch['pred_bi4s_pelvis_hm']).pow(2).sum((-1, -2))
    loss = loss.squeeze(1)
    loss_stats['pelvis_heatmap_MSE'] = loss.detach().cpu()
    return loss


def pelvis_depth_loss(trainer, batch, loss_stats):
    pred_bi4s_rdepth_hm = batch['pred_bi4s_rdepth_hm']
    gt_bi4s_pelvis_2d = batch['gt_bi4s_pelvis_2d']
    gkernel_3x3 = trainer.gkernel_3x3

    loss = []
    for b in range(pred_bi4s_rdepth_hm.size(0)):
        x, y = gt_bi4s_pelvis_2d[b]
        gt = batch['gt_bi_pelvis_zw1f'][b]
        pred = pred_bi4s_rdepth_hm[b, 0, y-1:y+2, x-1:x+2]
        loss.append(((gt - pred).abs() * gkernel_3x3).sum())
    loss = torch.stack(loss)
    loss_stats['pelvis_depth_L1'] = loss.detach().cpu()
    return loss


def bi4s_seg_hm_loss(trainer, batch, loss_stats):
    gt = batch['gt_bi4s_seg_hm']
    pred = batch['pred_bi4s_seg_hm']
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
    loss_stats['bi4s_seg_hm_MSE'] = loss.detach().cpu()
    return loss


def bi4s_seg_hm_flip_loss(trainer, batch, loss_stats):
    if not trainer.training:
        return torch.zeros_like(loss_stats['bi4s_seg_hm_MSE']).cuda()

    pred = batch['pred_bi4s_seg_hm_flip']
    gt = batch['gt_bi4s_seg_hm'].flip([-1])
    # need extra flip on label
    gt = gt[:, [1, 0, 3, 2, 4, 5, 6]]

    # MSE
    loss = (gt - pred).pow(2).sum((-1, -2))
    loss = loss.sum(-1)
    loss_stats['bi4s_seg_hm_flip_MSE'] = loss.detach().cpu()
    return loss


def pcSeg_voxel_loss(trainer, batch, loss_stats):
    zero_scalar = batch['image'].new([0.])
    label = batch['gt_vox_seg_label']
    pred = batch['pred_vox_seg']

    loss = []
    weight = torch.tensor([1.]*8).to(batch['image'])
    weight[0] *= trainer.loss_args.pcSeg_voxel_weight0  # balance background
    for b in range(len(pred)):
        if len(label[b]) > 0:
            loss.append(F.cross_entropy(pred[b], label[b], weight=weight, reduction='mean'))
            # cur_loss = F.cross_entropy(pred[b], label[b], reduction='none')
            # reweight = torch.ones_like(label[b]).float()
            # for i in range(8):
            #     if (label[b]==i).sum() > 0:
            #         reweight[label[b]==i] /= (label[b]==i).sum()
            # loss += (cur_loss * reweight).sum()
        else:
            loss.append(zero_scalar)
    loss = torch.stack(loss)
    loss_stats['pcSeg_voxel'] = loss.detach().cpu()
    return loss


def pcSeg_voxel_hm_loss(trainer, batch, loss_stats):
    gt = batch['gt_vox_seg_hm']
    pred = batch['pred_vox_seg_hm']
    loss = []
    for b in range(batch['B']):
        cur_loss = (gt[b] - pred[b]).pow(2).sum(-1).mean(-1)
        loss.append(cur_loss)
    loss = torch.stack(loss)
    loss_stats['pcSeg_voxel_hm'] = loss.detach().cpu()
    return loss


def pcSeg_voxel_focal_loss(trainer, batch, loss_stats):
    gamma = trainer.loss_args.pcSeg_voxel_focal_gamma
    zero_scalar = batch['image'].new([0.])
    label = batch['gt_vox_seg_label']
    pred = batch['pred_vox_seg']

    loss = []
    for b in range(len(pred)):
        if len(label[b]) > 0:
            ce_loss = F.cross_entropy(pred[b], label[b], reduction='none')  # N
            score = torch.softmax(pred[b], 1).gather(1, label[b][:, None]).squeeze(1)  # (N)
            cur_loss = ((1-score) ** gamma * ce_loss).mean()
            loss.append(cur_loss)
        else:
            loss.append(zero_scalar)
    loss = torch.stack(loss)
    loss_stats['pcSeg_voxel_focal'] = loss.detach().cpu()
    return loss


def depth_voxel_loss(trainer, batch, loss_stats):
    zero_scalar = batch['image'].new([0.])
    label = batch['gt_vox_depth_label']
    pred = batch['pred_vox_depth']

    loss = []
    for b in range(len(pred)):
        if len(label[b]) > 0:
            loss.append(F.binary_cross_entropy_with_logits(pred[b], label[b].float(), reduction='mean'))
            # cur_loss = F.binary_cross_entropy_with_logits(pred[b], label[b].float(), reduction='none')
            # loss += cur_loss[label[b]].sum() / (label[b].sum() + 1e-9) + \
            #     cur_loss[~label[b]].sum() / ((~label[b]).sum() + 1e-9)
        else:
            loss.append(zero_scalar)
    loss = torch.stack(loss)
    loss_stats['depth_voxel'] = loss.detach().cpu()
    return loss


def vox_z_offset_loss(trainer, batch, loss_stats):
    zero_scalar = batch['image'].new([0.])
    pred_vox_z_offset = [p[:, 0] for p in batch['pred_vox_xyz_offset']]  # historical reason
    gt_vox_z_offset = [gt[:, 2] for gt in batch['gt_vox_xyz_offset']]

    loss = []
    for b in range(len(pred_vox_z_offset)):
        if len(pred_vox_z_offset[b]) > 0:
            loss.append((pred_vox_z_offset[b] - gt_vox_z_offset[b]).abs().mean())
        else:
            loss.append(zero_scalar)
    loss = torch.stack(loss)
    loss_stats['vox_z_offset_L1'] = loss.detach().cpu()
    return loss


def vox_xyz_offset_loss(trainer, batch, loss_stats):
    zero_scalar = batch['image'].new([0.])
    pred_vox_xyz_offset = batch['pred_vox_xyz_offset']
    gt_vox_xyz_offset = batch['gt_vox_xyz_offset']

    loss = []
    for b in range(len(pred_vox_xyz_offset)):
        if len(pred_vox_xyz_offset[b]) > 0:
            loss.append((pred_vox_xyz_offset[b] - gt_vox_xyz_offset[b]).abs().sum(-1).mean())
        else:
            loss.append(zero_scalar)
    loss = torch.stack(loss)
    loss_stats['vox_xyz_offset_L1'] = loss.detach().cpu()
    return loss


def pelvis_depth_refined_loss(trainer, batch, loss_stats):
    pred_c_pelvis_refined = batch['pred_c_pelvis_refined']
    gt_c_pelvis = batch['gt_c_pelvis']

    loss = (pred_c_pelvis_refined[:, 0, 2] - gt_c_pelvis[:, 0, 2]).abs()
    loss_stats['pelvis_depth_refine_L1'] = loss.detach().cpu()
    return loss


def pelvis_xyz_refined_loss(trainer, batch, loss_stats):
    pred_c_pelvis_refined = batch['pred_c_pelvis_refined']
    gt_c_pelvis = batch['gt_c_pelvis']

    loss = (pred_c_pelvis_refined - gt_c_pelvis).abs().sum(-1).squeeze(1)
    loss_stats['pelvis_xyz_refined_L1'] = loss.detach().cpu()
    return loss


name2loss = {
    'vertex': vertex_loss,
    'vertex_c': vertex_c_loss,
    'joint2d': joint2d_loss,
    'joint': joint_loss,
    'joint_c': joint_c_loss,
    'vertex_contact_l1': vertex_contact_l1_loss,
    'vertex_contact_mse': vertex_contact_mse_loss,
    'corr_points_pred': corr_points_pred_loss,
    'corr_points_pred_immerse': corr_points_pred_immerse_loss,
    'feat_sim_cpv': feat_sim_cpv_loss,
    'corr': corr_loss,
    'pelvis_hm': pelvis_hm_loss,
    'pelvis_depth': pelvis_depth_loss,
    'bi4s_seg_hm': bi4s_seg_hm_loss,
    'bi4s_seg_hm_flip': bi4s_seg_hm_flip_loss,
    'pcSeg_voxel': pcSeg_voxel_loss,
    'pcSeg_voxel_focal': pcSeg_voxel_focal_loss,
    'pcSeg_voxel_hm': pcSeg_voxel_hm_loss,
    'depth_voxel': depth_voxel_loss,
    'vox_z_offset': vox_z_offset_loss,
    'vox_xyz_offset': vox_xyz_offset_loss,
    'pelvis_depth_refined': pelvis_depth_refined_loss,
    'pelvis_xyz_refined': pelvis_xyz_refined_loss,
    'vertex_offset': vertex_offset_loss,
    'vertex_refined': vertex_refined_loss,
    'vertex431_refined': vertex431_refined_loss,
    'vertex_cr_refined': vertex_cr_refined_loss,
    'joint_cr_refined': joint_cr_refined_loss,
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
