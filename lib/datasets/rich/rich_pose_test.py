from pathlib import Path
import numpy as np
import torch
import cv2
from torch.utils import data
from lib.utils import logger

from .rich_utils import extract_cam_xml


class Dataset(data.Dataset):
    """for test-set inference; also provides GT"""

    def __init__(self, rich_root="datasymlinks/RICH", **kwargs):
        super().__init__()
        # file path (original dataset)
        rich_root = Path(rich_root)

        # specially prepared data
        sahmr_support_dir = rich_root / "sahmr_support"
        self.scene_info_root = sahmr_support_dir / "scene_info"  # for cam-parameters
        self.sahmr_support_test_dir = sahmr_support_dir / "test_split"  # for test-split utils
        self.img_dir = self.sahmr_support_test_dir / "image"
        self.gt_mesh_dir = self.sahmr_support_test_dir / "gt_smplx_mesh"
        self.contact_root = self.sahmr_support_test_dir / "contact"

        # pre-load
        self.metas = np.load(self.sahmr_support_test_dir / "meta/test3116.npy", allow_pickle=True)
        self.cam2params = self._cam2params()  # cam_key -> (T_w2c, K)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        meta.update({"idx": idx, "data_name": "rich"})

        cf_ofn = meta["contact_file"]
        cf_fn = self.contact_root / cf_ofn[cf_ofn.find("test_contact/") + len("test_contact/") :]
        hsc_data = np.unpackbits(np.load(cf_fn), count=6890).reshape(-1, 1)
        T_w2c, K = self.cam2params[meta["cam_key"]]

        # process image with bbx, get new bbx
        img_key = meta["img_key"]
        img_squared = cv2.imread(str(self.img_dir / f"{img_key}.png"))[..., [2, 1, 0]]  # (224, 224, 3), in RGB, uint8
        img_squared = img_squared.transpose(2, 0, 1) / 255.0  # (3, H, W), float
        bbx_lurb_test_squared = meta["test_squared_bbx_lurb"]

        data = {
            "meta": meta,
            "gt_hsc": torch.from_numpy(hsc_data).float(),  # (6890, 1)
            "T_w2c": T_w2c.float(),  # (4, 4)
            "K": K.float(),  # (3, 3)
            "bbx_lurb": torch.from_numpy(bbx_lurb_test_squared).float(),  # (4,)
            "image": torch.from_numpy(img_squared).float(),  # (rgb, H, W)
        }

        return data

    def _cam2params(self):
        cam_params = {}
        cam_xml_files = self.scene_info_root.glob("*/calibration/*.xml")
        for cam_xml_file in cam_xml_files:
            cam_param = extract_cam_xml(cam_xml_file)
            T_w2c = cam_param["ext_mat"].reshape(3, 4)
            T_w2c = torch.cat([T_w2c, torch.tensor([[0, 0, 0, 1.0]])], dim=0)  # (4, 4)
            K = cam_param["int_mat"].reshape(3, 3)
            cap_name = cam_xml_file.parts[-3]
            cam_id = int(cam_xml_file.stem)
            cam_key = f"{cap_name}_{cam_id}"
            cam_params[cam_key] = (T_w2c, K)
        return cam_params
