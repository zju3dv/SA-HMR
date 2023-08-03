import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from pathlib import Path
from lib.utils import logger
from lib.train.trainers.pose.loss import name2loss
from lib.networks.pose.sahmr._smpl import Mesh as mesh_sampler
from lib.networks.pose.sahmr._smpl import SMPL
from lib.utils.smplx_utils import make_smplx

from lib.utils.geo_transform import apply_T_on_points
from lib.train.trainers.pose.supervision import *
from lib.train.trainers.pose.preprocess import *


class NetworkWrapper(nn.Module):
    def __init__(self, net: nn.Module, loss_weights, data_name, **kwargs):
        super().__init__()
        self.net = net  # main network
        self.data_name = data_name

        # config@train
        self.train_extra_flip = kwargs["train_extra_flip"]  # when training rcnet, set to True
        self.loss_weights = {k: v for k, v in loss_weights.items() if v > 0}

        # config@spv
        self.contact_thr_dist = 0.07
        self.skip_spv_bi4s_seg = kwargs["skip_spv_bi4s_seg"]
        self.register_buffer(f"gkernel_3x3", gkernel_3x3, False)

        # helper variables
        self.smpl = SMPL()
        self.mesh_sampler = mesh_sampler()
        self.smpl_face = self.mesh_sampler.faces.numpy()

        init_verts, init_joints14 = self.smpl.initialize_h36m_verts_and_joints14()
        init_verts1723 = self.mesh_sampler.downsample(init_verts)
        init_verts431 = self.mesh_sampler.downsample(init_verts1723, n1=1, n2=2)  # (1, 431, 3)
        self.register_buffer("init_verts431", init_verts431, False)
        self.register_buffer("init_joints14", init_joints14, False)

        seg_part_info = np.load("lib/utils/body_model/seg_part_info.npy", allow_pickle=True).item()
        verts6890_pids = torch.from_numpy(seg_part_info["verts6890_pids"]).long()
        verts431_pids = torch.from_numpy(seg_part_info["verts431_pids"]).long()
        self.register_buffer("verts6890_pids", verts6890_pids, False)
        self.register_buffer("verts431_pids", verts431_pids, False)
        for i in range(1, 8):
            self.register_buffer(f"pid_to_verts431_{i}", torch.where(self.verts431_pids == i)[0], False)

        # Scene information
        if data_name == "rich":
            self.setup_rich()
        elif data_name == "prox":
            raise NotImplementedError
        elif data_name == "prox_quant":
            self.setup_prox_quant()

    def setup_rich(self):
        scan_root = Path("datasymlinks/RICH/sahmr_support/scene_info")
        voxel_files = list(scan_root.glob("*/*_world.npy"))  # _world in w; without _world in az
        self.voxel_pitchs = {}
        for file in voxel_files:
            scene_key = file.parts[-2] + "_" + file.stem[: file.stem.find("camcoord") + 8]  # fix w/wo _world
            data = np.load(file, allow_pickle=True).tolist()
            self.register_buffer(f"{scene_key}_voxel", torch.from_numpy(data["points"]).float(), False)
            self.voxel_pitchs[scene_key] = data["pitch"][0]

        # supervision
        self.male_smplh = make_smplx("rich-smplh", gender="male")
        self.female_smplh = make_smplx("rich-smplh", gender="female")
        self.smplh = {"male": self.male_smplh, "female": self.female_smplh}

    def setup_prox_quant(self):
        if not hasattr(self, "voxel_pitchs"):
            self.voxel_pitchs = {}

        ("voxel_pitchs", dict())
        # quantitative
        vicon_folder = "datasymlinks/PROX/quantitative/sahmr_support/vicon"
        scene_key = "vicon"
        data = np.load(f"{vicon_folder}/vicon_voxel_world.npy", allow_pickle=True).tolist()
        self.register_buffer(f"{scene_key}_voxel", torch.from_numpy(data["points"]).float(), False)
        self.voxel_pitchs[scene_key] = data["pitch"][0]

        # supervision (smplx -> smplh deformation matrix)
        smplx2smplh_def_pth = Path("models/model_transfer/smplx2smplh_deftrafo_setup.pkl")
        smplx2smplh_def = pickle.load(smplx2smplh_def_pth.open("rb"), encoding="latin1")
        smplx2smplh_def = np.array(smplx2smplh_def["mtx"].todense(), dtype=np.float32)[:, :10475]  # (6890, 10475)
        self.register_buffer(f"smplx2smplh_def", torch.from_numpy(smplx2smplh_def).float(), False)

    def forward(self, batch, compute_supervision=True, compute_loss=True):
        self.preprocess(batch)
        self.net(batch)

        if compute_supervision:
            self.compute_supervision(batch)
        if compute_loss:
            self.compute_loss(batch)

        return batch  # important for DDP

    def preprocess(self, batch):
        B, _, H, W = batch["image"].shape
        assert W == H, "we assume a squared input image"
        assert W % 4 == 0
        batch["B"] = B
        batch["smpl"] = self.smpl
        batch["mesh_sampler"] = self.mesh_sampler
        batch["init_joints14"] = self.init_joints14.expand(B, -1, -1).contiguous()
        batch["init_verts431"] = self.init_verts431.expand(B, -1, -1).contiguous()
        batch["verts6890_pids"] = self.verts6890_pids
        batch["verts431_pids"] = self.verts431_pids
        batch["pid_to_verts431"] = {i: self.__getattr__(f"pid_to_verts431_{i}") for i in range(1, 8)}
        batch["ri_f"] = batch["K"][:, 0, 0]
        batch["bi_f"] = batch["ri_f"] * W / (batch["bbx_lurb"][:, 2] - batch["bbx_lurb"][:, 0])
        batch["bi_w"] = W
        batch["flag_train_extra_flip"] = self.train_extra_flip

        get_pc_in_frustum(self, batch)

        if self.training:
            batch["gt_w_verts"] = get_gt_smpl_verts_in_world(self, batch)
            batch["gt_c_verts"] = apply_T_on_points(batch["gt_w_verts"], batch["T_w2c"])
            batch["gt_c_pelvis"] = self.smpl.get_h36m_pelvis(batch["gt_c_verts"])

    def compute_supervision(self, batch):
        """use functions from ./supervision.py"""
        # gt Verts and Joints in the world and cam coordinates: (B, 6890, 3), (B, 14, 3)
        if "gt_w_verts" not in batch:
            batch["gt_w_verts"] = get_gt_smpl_verts_in_world(self, batch)
        if "gt_w_joints14" not in batch:
            batch["gt_w_joints14"] = self.smpl.get_h36m_joints14(batch["gt_w_verts"])
        if "gt_c_verts" not in batch:
            batch["gt_c_verts"] = apply_T_on_points(batch["gt_w_verts"], batch["T_w2c"])
        if "gt_c_joints14" not in batch:
            batch["gt_c_joints14"] = apply_T_on_points(batch["gt_w_joints14"], batch["T_w2c"])
        if "gt_c_pelvis" not in batch:
            batch["gt_c_pelvis"] = self.smpl.get_h36m_pelvis(batch["gt_c_verts"])
        if "gt_c_verts431" not in batch:
            batch["gt_c_verts431"] = batch["mesh_sampler"].downsample_to_verts431(batch["gt_c_verts"])

        # spv on verts and joints14 (regularize)
        if "gt_cr_verts" not in batch or "gt_cr_joints14" not in batch:
            batch["gt_cr_verts"], batch["gt_cr_joints14"] = self.smpl.get_r_h36m_verts_joints14(batch["gt_c_verts"])

        # spv on bbx img (2d keypoints)
        if "gt_bi_joints14_2d" not in batch:
            batch["gt_bi_joints14_2d"] = get_gt_joints2d_in_bbx_img(batch)

        # spv on pelvis, rdepth, and pcSeg
        spv_pelvis_position(self, batch)
        spv_pcSeg(self, batch)
        spv_pelvis_refine(self, batch)  # call this after spv_pcSeg
        spv_bi4s_seg(self, batch)  # auxiliary supervision
        return

    def compute_loss(self, batch):
        # """use functions from ./loss.py"""
        B = batch["B"]
        loss = 0.0
        loss_stats = {}
        for k, v in self.loss_weights.items():
            cur_loss = v * name2loss[k](self, batch, loss_stats)
            assert cur_loss.shape[0] == B
            loss += cur_loss
        loss_stats.update({"loss_weighted_sum": loss.detach().cpu()})
        batch.update({"loss": loss, "loss_stats": loss_stats})
