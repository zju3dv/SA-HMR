import time
import torch
import torch.nn.functional as F
import pytorch3d.ops.knn as knn
from lib.utils.geo_transform import project_p2d, apply_T_on_points, cvt_to_bi01_p2d, get_nearby_points


gkernel_3x3 = torch.tensor([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]])


def get_w_pcSeg_all(trainer, batch, thr_dist=0.05, skip_hand=False):
    """Find pcs that near(belong) to vertsSeg
    Returns: List, List
    """
    B = batch["gt_w_verts"].size(0)
    verts6890_seg_mask = batch["verts6890_pids"] != 0  # to limit query range
    if skip_hand:
        hand_mask = (batch["verts6890_pids"] == 3) + (batch["verts6890_pids"] == 4)
        verts6890_seg_mask = verts6890_seg_mask * ~hand_mask
    vertsSeg_pids = batch["verts6890_pids"][verts6890_seg_mask]  # verts{num(Seg)}

    pcSeg_all, pcSeg_all_pids = [], []
    for b in range(B):
        # 1. all scene point
        scene_key = batch["meta"][b]["scene_key"]
        pitch = trainer.voxel_pitchs[scene_key]
        if "w_pcFst_all" in batch:
            pc_w = batch["w_pcFst_all"][b]  # points in frustum
        else:
            pc_w = trainer.__getattr__(f"{scene_key}_voxel")  # all scene points

        # 2. delete points outside 3d_bbx
        query_verts = batch["gt_w_verts"][b][verts6890_seg_mask]  # (P, 3)
        pc_w = get_nearby_points(pc_w, query_verts, padding=thr_dist + 0.05)  # (S, 3) 0.05 is a safe-bound
        pc_w += (torch.rand_like(pc_w) * 2 - 1) * pitch * 0.3

        # 3. delete points farther than thr_dist
        dist_mat = torch.norm(query_verts[:, None] - pc_w[None], 2, -1)  # L2, (P, S)
        pc_w_kept_mask = dist_mat.min(0)[0] < thr_dist

        #  4. record point position and the its part-id
        pcSeg_all.append(pc_w[pc_w_kept_mask])  # might by empty
        indices = dist_mat[:, pc_w_kept_mask].min(0)[1]
        pcSeg_all_pids.append(vertsSeg_pids[indices])

    return pcSeg_all, pcSeg_all_pids


@torch.no_grad()
def get_gt_smpl_verts_in_world(trainer, batch):
    if trainer.cfg.dataset_name == "prox":
        if "gt_smplx_params" in batch:  # proxdfitting
            smplh_opt = {
                "male": trainer.smplx["male"](**batch["gt_smplx_params"]).vertices,
                "female": trainer.smplx["female"](**batch["gt_smplx_params"]).vertices,
            }
            gt_c_verts = torch.stack([smplh_opt[meta["gender"]][i] for i, meta in enumerate(batch["meta"])])
            gt_c_verts = trainer.smplx2smplh_def[None] @ gt_c_verts
            gt_w_verts = apply_T_on_points(gt_c_verts, batch["T_c2w"])

        elif "gt_smplh_params" in batch:
            if trainer.prox_annot == "humor_fitting":
                gt_c_verts = trainer.smplh(**batch["gt_smplh_params"]).v
            else:
                gt_c_verts = trainer.smplh(**batch["gt_smplh_params"]).vertices
            gt_w_verts = apply_T_on_points(gt_c_verts, batch["T_c2w"])
        else:  # mesh
            gt_c_verts = trainer.smplx2smplh_def[None] @ batch["gt_c_smplx_verts"]
            gt_w_verts = apply_T_on_points(gt_c_verts, batch["T_c2w"])

    elif trainer.cfg.dataset_name == "rich":
        smplh_opt = {
            "male": trainer.smplh["male"](**batch["gt_smplh_params"]).vertices,
            "female": trainer.smplh["female"](**batch["gt_smplh_params"]).vertices,
        }
        gt_w_verts = torch.stack([smplh_opt[meta["gender"]][i] for i, meta in enumerate(batch["meta"])])
    return gt_w_verts


def get_gt_joints2d_in_bbx_img(batch):
    # raw img
    gt_ri_joints14_2d = project_p2d(batch["gt_c_joints14"], batch["K"])

    # bbx img
    gt_bi_joints14_2d = gt_ri_joints14_2d - batch["bbx_lurb"][:, None, :2]
    gt_bi_joints14_2d[..., 0] = (
        gt_bi_joints14_2d[..., 0]
        * (batch["image"].size(3) - 1)
        / (batch["bbx_lurb"][:, 2] - batch["bbx_lurb"][:, 0])[:, None]
    )
    gt_bi_joints14_2d[..., 1] = (
        gt_bi_joints14_2d[..., 1]
        * (batch["image"].size(2) - 1)
        / (batch["bbx_lurb"][:, 3] - batch["bbx_lurb"][:, 1])[:, None]
    )

    return gt_bi_joints14_2d


def spv_pelvis_position(trainer, batch):
    assert "gt_c_verts" in batch
    if "gt_c_pelvis" not in batch:
        batch["gt_c_pelvis"] = trainer.smpl.get_h36m_pelvis(batch["gt_c_verts"])
    B = batch["gt_c_pelvis"].size(0)

    # raw img
    gt_ri_pelvis_2d = project_p2d(batch["gt_c_pelvis"], batch["K"])[:, 0]  # (B, 2)

    # bbx img
    gt_bi01_pelvis_2d = gt_ri_pelvis_2d - batch["bbx_lurb"][:, :2]
    gt_bi01_pelvis_2d /= (batch["bbx_lurb"][:, 2] - batch["bbx_lurb"][:, 0])[:, None]
    batch["gt_bi01_pelvis_2d"] = gt_bi01_pelvis_2d  # (B, 2)

    # create heatmap
    gt_bi4s_pelvis_hm = torch.zeros_like(batch["pred_bi4s_pelvis_hm"])  # (B, 1, H/4, W/4)
    max_size = gt_bi4s_pelvis_hm.size(2)
    gt_bi4s_pelvis_2d = (gt_bi01_pelvis_2d * (max_size - 1)).round().int()
    for b in range(B):
        x, y = gt_bi4s_pelvis_2d[b]
        x = 1 if x < 1 else x
        y = 1 if y < 1 else y
        x = max_size - 2 if x >= max_size - 1 else x
        y = max_size - 2 if y >= max_size - 1 else y
        gt_bi4s_pelvis_hm[b, 0, y - 1 : y + 2, x - 1 : x + 2] = trainer.gkernel_3x3
    batch["gt_bi4s_pelvis_hm"] = gt_bi4s_pelvis_hm
    batch["gt_bi4s_pelvis_2d"] = gt_bi4s_pelvis_2d

    # root depth (normalized by FoV)
    batch["gt_c_pelvis_z"] = gt_c_pelvis_z = batch["gt_c_pelvis"][:, 0, 2]
    batch["gt_bi_pelvis_zw1f"] = gt_c_pelvis_z * batch["bi_w"] / batch["bi_f"]


@torch.no_grad()
def spv_bi4s_seg(trainer, batch):
    if trainer.skip_spv_bi4s_seg:
        return

    tic = time.time()
    B = batch["image"].size(0)
    assert "c_pcFst_all" in batch and "w_pcFst_all" in batch

    # 1. calculate all w_pcSeg on-the-fly
    if "gt_w_pcSeg_all" not in batch:
        skip_hand = batch["meta"][0]["data_name"] in ["prox", "prox_quant"]
        batch["gt_w_pcSeg_all"], batch["gt_pcSeg_all_pids"] = get_w_pcSeg_all(
            trainer, batch, trainer.contact_thr_dist, skip_hand
        )

    w_pcSeg_all, pcSeg_all_pids = batch["gt_w_pcSeg_all"], batch["gt_pcSeg_all_pids"]
    c_pcSeg_all = [apply_T_on_points(p[None], batch["T_w2c"][[b]])[0] for b, p in enumerate(w_pcSeg_all)]
    bi01_pcSeg_all_2d = [
        cvt_to_bi01_p2d(project_p2d(p[None], batch["K"][[b]]), batch["bbx_lurb"][[b]])[0]
        for b, p in enumerate(c_pcSeg_all)
    ]
    batch["gt_c_pcSeg_all"] = c_pcSeg_all  # [(N, 2)]
    batch["gt_bi01_pcSeg_all_2d"] = bi01_pcSeg_all_2d  # [(N, 2)]

    gt_bi4s_seg_hm = torch.zeros_like(batch["pred_bi4s_seg_hm"])  # (B, 7, H/4, W/4)
    W = gt_bi4s_seg_hm.size(-1)

    bpxy = []
    for b in range(B):
        mask = pcSeg_all_pids[b] != 0
        p_ = (pcSeg_all_pids[b][mask] - 1)[:, None]  # (N, 1) put to 7 class, not pid==0 is background
        xy_ = (bi01_pcSeg_all_2d[b][mask] * (W - 1)).round().long()  # (N, 2)
        mask_inside = (xy_ > -1).all(-1) * (xy_ < W).all(-1)
        p_, xy_ = p_[mask_inside], xy_[mask_inside]
        bpxy_ = F.pad(torch.cat([p_, xy_], dim=1), (1, 0), value=b)
        bpxy_ = torch.unique(bpxy_, dim=0)
        bpxy.append(bpxy_.reshape(-1, 4))  # (N, 4)
    bpxy = torch.cat(bpxy, dim=0)
    # set those point to 1
    gt_bi4s_seg_hm[bpxy[:, 0], bpxy[:, 1], bpxy[:, 3], bpxy[:, 2]] = 1
    # gaussian blur
    conv_weight = trainer.gkernel_3x3.reshape(1, 1, 3, 3).repeat(7, 1, 1, 1)
    gt_bi4s_seg_hm = F.conv2d(gt_bi4s_seg_hm, conv_weight, padding=1, groups=7)
    gt_bi4s_seg_hm.clamp_max_(1.0)

    batch["gt_bi4s_seg_hm"] = gt_bi4s_seg_hm  # (B, 7, H/4, W/4)


@torch.no_grad()
def spv_pcSeg(trainer, batch):
    B = batch["image"].size(0)

    # 1. Find contact voxels
    verts6890_seg_mask = batch["verts6890_pids"] != 0  # to limit query range
    skip_hand = batch["meta"][0]["data_name"] in ["prox", "prox_quant"]
    if skip_hand:
        hand_mask = (batch["verts6890_pids"] == 3) + (batch["verts6890_pids"] == 4)
        verts6890_seg_mask = verts6890_seg_mask * ~hand_mask
    vertsSeg_pids = batch["verts6890_pids"][verts6890_seg_mask]  # verts{num(Seg)}
    c_voxFst_nearby = batch["c_voxFst_nearby"]

    thr_dist = trainer.contact_thr_dist  # a far-far-away version I use 0.075cm, current default 0.05

    gt_vox_seg_label = [torch.zeros_like(x[:, 0]).long() for x in c_voxFst_nearby]
    for b in range(B):
        # find contact voxel point (< thr_dist)
        query_verts = batch["gt_c_verts"][b][verts6890_seg_mask]  # (V, 3)
        ref_xyz = c_voxFst_nearby[b]  # (N, 3)
        dist_mat = (query_verts[:, None] - ref_xyz[None]).norm(p=2, dim=-1)  # (V, N)
        vox_contact_mask = dist_mat.min(0)[0] < thr_dist

        #  record point position and the its part-id
        gt_vox_seg_label[b][vox_contact_mask] = vertsSeg_pids[dist_mat[:, vox_contact_mask].min(0)[1]]

    batch["gt_vox_seg_label"] = gt_vox_seg_label


def spv_pelvis_refine(trainer, batch):
    """call this after spv_pcSeg"""
    c_voxFst_nearby = batch["c_voxFst_nearby"]  # [(N, 3)]
    gt_c_pelvis = batch["gt_c_pelvis"]  # (B, 1, 3)
    gt_vox_xyz_offset = [gt_c_pelvis[b] - c_voxFst_nearby[b] for b in range(batch["B"])]
    batch["gt_vox_xyz_offset"] = gt_vox_xyz_offset  # [(N, 3)]
