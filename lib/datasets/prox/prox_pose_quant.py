from pathlib import Path
import numpy as np
import pickle
import json
import torch
import cv2
from torch.utils import data
from lib.datasets.prox.prox_utils import squared_crop_and_resize
import trimesh


class Dataset(data.Dataset):
    def __init__(self, prox_root="datasymlinks/PROX/quantitative", **kwargs):
        super().__init__()
        # file path (original dataset)
        prox_root = Path(prox_root)

        # specially prepared data
        sahmr_support_dir = prox_root / "sahmr_support"
        self.sahmr_support_dir = sahmr_support_dir
        meta_fn = sahmr_support_dir / "meta/mosh_mesh_test.pkl"
        img2gtbbx_fn = sahmr_support_dir / f"img2gtbbx/mosh_mesh.pkl"
        self.mosh_mesh_dir = sahmr_support_dir / "gt_mosh_mesh"
        self.image_dir = sahmr_support_dir / "image"

        # load all data available
        self.metas = pickle.load(open(str(meta_fn), "rb"))  # list
        self.img2gtbbx = pickle.load(open(str(img2gtbbx_fn), "rb"))  # img_key -> bbx_lurb
        self.K = self.get_K()  # torch.Tesnor: (3, 3)
        self.T_c2w = self.get_T_c2w()  # torch.Tesnor: (4, 4)
        self.T_w2c = self.T_c2w.inverse()

        # disable bbx augmentation
        self.BBX_CENTER = self.BBX_ZOOM = 0

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx].copy()
        meta.update({"data_name": "prox_quant", "scene_key": meta["scene_name"]})

        bf_ofn = meta["body_file"]
        bf_fn = self.mosh_mesh_dir / bf_ofn[bf_ofn.find("mosh/") + len("mosh/") :]
        meta['body_file'] = str(bf_fn)
        gt_c_smplx_mesh = trimesh.load(bf_fn, process=False)  # 10475
        gt_c_smplx_verts = torch.from_numpy(gt_c_smplx_mesh.vertices).float()
        data = {
            "meta": meta,
            "gt_c_smplx_verts": gt_c_smplx_verts,  # (10475, 3)
            "T_c2w": self.T_c2w.clone(),  # (4, 4)
            "T_w2c": self.T_w2c.clone(),  # (4, 4)
            "K": self.K.clone(),  # (3, 3)
        }

        # process image with bbx, get new bbx
        bbx_lurb_human = self.img2gtbbx[meta["img_key"]]
        img_ofn = meta["img_file"]
        img_fn = self.image_dir / img_ofn[img_ofn.find("recordings/") + len("recordings/") :]
        meta['img_file'] = str(img_fn)
        img = cv2.imread(str(img_fn))[..., [2, 1, 0]]  # (H, W, 3), in RGB, uint8, , (1080, 1920, 3)
        img = cv2.flip(img, 1)
        img, bbx_lurb, A = squared_crop_and_resize(self, img, bbx_lurb_human)
        img = img.transpose(2, 0, 1) / 255.0  # (3, H, W), float

        data.update(
            {
                "bbx_lurb_human": torch.from_numpy(bbx_lurb_human),  # (4,)
                "bbx_lurb": torch.from_numpy(bbx_lurb).float(),  # (4,)
                "image": torch.from_numpy(img).float(),  # (rgb, H, W)
            }
        )

        return data

    def get_K(self):
        file = self.sahmr_support_dir / "vicon/calibration/Color.json"
        data = json.load(file.open("r"))
        K = np.zeros((3, 3))
        K[0, 2] = data["c"][0]
        K[1, 2] = data["c"][1]
        K[0, 0] = data["f"][0]
        K[1, 1] = data["f"][1]
        return torch.from_numpy(K).float()

    def get_T_c2w(self):
        file = self.sahmr_support_dir / "vicon/calibration/cam2world_vicon.json"
        with open(file, "r") as f:
            T_c2w = np.array(json.load(f))
        return torch.from_numpy(T_c2w).float()
