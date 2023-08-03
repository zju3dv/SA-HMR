import argparse
import pickle
import json
import trimesh
from time import time
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib.config import make_cfg
from lib.utils import logger
from lib.utils.net_utils import to_cuda
from lib.datasets.rich.rich_pose_test import Dataset as RICH_Dataset
from lib.datasets.prox.prox_pose_quant import Dataset as PROX_quant_Dataset
from lib.datasets.make_dataset import collate_fn_wrapper
from lib.utils.geo_transform import apply_T_on_points


def sub_glob(dir, pattern):
    len_dir = len(str(dir))
    return [str(p)[len_dir + 1 :] for p in dir.glob(pattern)]


def L2_error(x, y):
    return (x - y).pow(2).sum(-1).sqrt()


def to_np(x):
    return x.cpu().numpy()


def load_scene_sdfs():  # RICH
    # pre-load sdf
    sahmr_support_scene_dir = Path('datasymlinks/RICH/sahmr_support/scene_info')
    scanids = [
        "ParkingLot2/scan_camcoord",
        "LectureHall/scan_chair_scene_camcoord",
        "LectureHall/scan_yoga_scene_camcoord",
        "Gym/scan_camcoord",
        "Gym/scan_table_camcoord",
    ]

    tic = time()
    scanid_2_sdf = {}
    sdf_json_fns = [str(sahmr_support_scene_dir / f"{f}_mysdf.json") for f in scanids]
    sdf_fns = [str(sahmr_support_scene_dir / f"{f}_mysdf.npy") for f in scanids]
    for scanid, json_fn, sdf_fn in zip(scanids, sdf_json_fns, sdf_fns):
        with open(json_fn, "r") as f:
            sdf_data = json.load(f)
            grid_min = torch.tensor(np.array(sdf_data["min"])).float().cuda()
            grid_max = torch.tensor(np.array(sdf_data["max"])).float().cuda()
            grid_dim = sdf_data["dim"]
            voxel_size = (grid_max - grid_min) / grid_dim
        sdf = torch.tensor(np.load(sdf_fn).reshape(grid_dim, grid_dim, grid_dim)).float().cuda()
        scanid_2_sdf[scanid] = (sdf, grid_min, grid_max, grid_dim)
    print("Pre-Load mesh and sdf:", time() - tic)
    return scanid_2_sdf


def load_prox_scene_sdfs():  # PROX_quant
    sahmr_support_scene_dir = Path("datasymlinks/PROX/quantitative/sahmr_support/vicon")
    scanids = ["vicon"]

    tic = time()
    scanid_2_sdf = {}
    sdf_json_fns = [str(sahmr_support_scene_dir / f"{f}.json") for f in scanids]
    sdf_fns = [str(sahmr_support_scene_dir / f"{f}_sdf.npy") for f in scanids]
    for scanid, json_fn, sdf_fn in zip(scanids, sdf_json_fns, sdf_fns):
        with open(json_fn, "r") as f:
            sdf_data = json.load(f)
            grid_min = torch.tensor(np.array(sdf_data["min"])).float().cuda()
            grid_max = torch.tensor(np.array(sdf_data["max"])).float().cuda()
            grid_dim = sdf_data["dim"]
            voxel_size = (grid_max - grid_min) / grid_dim
        sdf = torch.tensor(np.load(sdf_fn).reshape(grid_dim, grid_dim, grid_dim)).float().cuda()
        scanid_2_sdf[scanid] = (sdf, grid_min, grid_max, grid_dim)
    print("Pre-Load mesh and sdf:", time() - tic)
    return scanid_2_sdf


def main(cfg):
    data_name = cfg.data_name

    # check if pred_dump exists
    out_dir = Path(f"out/pred_dump/{cfg.task}")
    assert out_dir.exists(), "Run 'python tools/dump_results.py' first"

    # data
    if data_name == "rich":
        dataset = RICH_Dataset()
        data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_wrapper)
        gt_dir = Path("datasymlinks/RICH/sahmr_support/test_split/gt_smplx_mesh")
        sdfs = load_scene_sdfs()
    elif data_name == "prox_quant":
        dataset = PROX_quant_Dataset()
        data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_wrapper)
        sdfs = load_prox_scene_sdfs()

    # 2. The prediction uses SMPLH whereas the groundtruths are SMPLX.
    # We need to convert SMPLX to SMPLH to calculate the MPJPE/MPVE/PenE
    smplx2smplh_def_pth = Path("models/model_transfer/smplx2smplh_deftrafo_setup.pkl")
    smplx2smplh_def = pickle.load(smplx2smplh_def_pth.open("rb"), encoding="latin1")
    smplx2smplh_def = np.array(smplx2smplh_def["mtx"].todense(), dtype=np.float32)[:, :10475]  # (6890, 10475)
    smplx2smplh_def = torch.from_numpy(smplx2smplh_def).float().cuda()

    H36M_J_regressor = np.load(
        "lib/networks/pose/sahmr/data/J_regressor_h36m_correct.npy", allow_pickle=True
    )  # (17, 6890)
    H36M_J_regressor = torch.from_numpy(H36M_J_regressor).float().cuda()
    H36M_J17_TO_J14 = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12, 13, 8, 10]

    GMPJPEs, GMPVEs, MPJPEs, MPVEs, PenEs, ConFEs = [], [], [], [], [], []
    for batch in tqdm(data_loader):
        batch = to_cuda(batch)

        meta = batch["meta"][0]
        img_key = meta["img_key"]
        pred_path = out_dir / f"{img_key}.pkl"

        if data_name == "rich":
            # path
            gt_path = gt_dir / f"{img_key}.ply"

            # scene + T_c2w
            sdf, grid_min, grid_max, grid_dim = sdfs[f"{meta['cap_name']}/{meta['scan_name']}"]
            T_c2w = batch["T_w2c"].inverse()[0]  # [4,4]

            # load human-scene contact mask
            hsc_mask = batch["gt_hsc"][0, :, 0].bool()  # (6890,) bool

        elif data_name == "prox_quant":
            # path
            gt_path = meta["body_file"]

            # scene + T_c2w
            sdf, grid_min, grid_max, grid_dim = sdfs[f"{meta['scene_key']}"]
            T_c2w = batch["T_c2w"][0]  # [4,4]

            # load human-scene contact mask
            hsc_mask = torch.zeros(6890).bool().cuda()  # (6890,) bool

        # load gt
        gt_mesh = trimesh.load(gt_path, process=False)  # (10475, 3), in SMPLX
        gt_verts = torch.from_numpy(gt_mesh.vertices).float().cuda()
        gt_verts = smplx2smplh_def @ gt_verts
        if data_name == "prox_quant":
            gt_verts = apply_T_on_points(gt_verts[None], T_c2w[None])[0]  # (6890, 3)

        # load pred
        with (pred_path).open("rb") as f:
            dict_pred = pickle.load(f)
        pred_verts = torch.from_numpy(dict_pred["pred_c_verts"]).cuda()
        pred_verts = apply_T_on_points(pred_verts[None], T_c2w[None])[0]  # (6890, 3)

        # ----- #
        gt_joints = H36M_J_regressor @ gt_verts
        gt_pelvis = gt_joints[0]
        gt_joints = gt_joints[H36M_J17_TO_J14]

        pred_joints = H36M_J_regressor @ pred_verts
        pred_pelvis = pred_joints[0]
        pred_joints = pred_joints[H36M_J17_TO_J14]

        q_gs_points = ((pred_verts - grid_min) / (grid_max - grid_min) * 2 - 1)[None]
        q_gs_sdfs = F.grid_sample(
            sdf.view(1, 1, grid_dim, grid_dim, grid_dim),
            q_gs_points[:, :, [2, 1, 0]].view(1, 1, 1, 6890, 3),
            padding_mode="border",
            align_corners=True,
        ).reshape(-1)
        PenE = q_gs_sdfs.abs() * (q_gs_sdfs < 0).float()
        ConFE = torch.cat([PenE[~hsc_mask], q_gs_sdfs.abs()[hsc_mask]])

        PenEs.append(to_np(PenE.sum()))
        ConFEs.append(to_np(ConFE.sum()))
        GMPJPEs.append(to_np(L2_error(gt_joints, pred_joints).mean() * 1000))
        GMPVEs.append(to_np(L2_error(gt_verts, pred_verts).mean() * 1000))
        MPJPEs.append(to_np(L2_error(gt_joints, pred_joints - pred_pelvis + gt_pelvis).mean() * 1000))
        MPVEs.append(to_np(L2_error(gt_verts, pred_verts - pred_pelvis + gt_pelvis).mean() * 1000))

    summarized_metrics = {
        "GMPJPE": np.mean(GMPJPEs),
        "GMPVE": np.mean(GMPVEs),
        "MPJPE": np.mean(MPJPEs),
        "MPVE": np.mean(MPVEs),
        "PenE": np.mean(PenEs),
        "ConFE": np.mean(ConFEs),
    }
    for k, v in summarized_metrics.items():
        print(f"{k}: {v:.2f}")
    print("---------- ⬆️ ----------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", "-c", type=str, required=True)

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = make_cfg(args)

    main(cfg)
