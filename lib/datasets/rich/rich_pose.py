from pathlib import Path
import numpy as np
import pickle
import torch
import cv2
from torch.utils import data
from lib.utils import logger

from .rich_utils import (
    get_cam2params,
    remove_extra_rules,
    squared_crop_and_resize,
    sample_idx2meta,
    remove_bbx_invisible_frame,
)


class Dataset(data.Dataset):
    def __init__(
        self,
        rich_root="datasymlinks/RICH",
        split="train",
        sample_interval=1,
        bbx_center=10,
        bbx_zoom=0.4,
        **kwargs,
    ):
        super().__init__()
        assert split in ["train", "val"]
        self.split = split

        self.PROJ_SCALE = 4  # (4112, 3008) / (1028, 752) = 4
        # bbx augmentation
        self.BBX_CENTER = bbx_center  # center + noise in (center +/- BBX_CENTER)
        self.BBX_ZOOM = bbx_zoom  # half size * range in (1, 1 + 0.15 + BBX_ZOOM)

        # file path (original dataset, but all preprocessed)
        rich_root = Path(rich_root)
        self.img_root = rich_root / "images_ds4" / self.split  # compressed (not original)
        self.body_root = rich_root / "bodies" / self.split  # we fit smplh to each smplx

        # specially prepared data
        sahmr_support_dir = rich_root / "sahmr_support"
        self.scene_info_root = sahmr_support_dir / "scene_info"  # for cam-parameters
        split_root = sahmr_support_dir / f"{self.split}_split"
        self.meta_root = split_root / "meta"  # highly-compressed
        self.contact_root = split_root / "contact"  # highly-compressed

        # pre-load
        self.cam2params = get_cam2params(self.scene_info_root)  # cam_key -> (T_w2c, K)
        self.img2gtbbx = self._img2gtbbx()  # img_key -> bbx_lurb
        self.idx2meta = sample_idx2meta(self._idx2meta(), sample_interval)  # meta, List
        self.idx2meta = remove_extra_rules(self.idx2meta)
        self.idx2meta = remove_bbx_invisible_frame(self.idx2meta, self.img2gtbbx)

    def __len__(self):
        return len(self.idx2meta)

    def __getitem__(self, idx):
        # The meta contains some absolute paths, which are updated in the following code.
        meta = self.idx2meta[idx]
        meta.update({"idx": idx, "data_name": "rich"})

        new_body_file = self.body_root / "/".join(meta["body_file"].split("/")[-3:])
        data = pickle.load(open(new_body_file, "rb"))
        smplh_params = {k: torch.from_numpy(v).float() for k, v in data.items()}

        new_contact_file = self.contact_root / "/".join(meta["contact_file"].split("/")[-3:])
        data_packed = np.load(new_contact_file)
        hsc_data = np.unpackbits(data_packed, count=6890).reshape(-1, 1)
        T_w2c, K = self.cam2params[meta["cam_key"]]
        data = {
            "meta": meta,
            "gt_smplh_params": smplh_params,
            "gt_hsc": torch.from_numpy(hsc_data).float(),  # (6890, 1)
            "T_w2c": T_w2c.float(),  # (4, 4)
            "K": K.float(),  # (3, 3)
        }

        # process image with bbx, get new bbx
        bbx_lurb_human = self.img2gtbbx[meta["img_key"]]
        bbx_lurb_resized = bbx_lurb_human / self.PROJ_SCALE
        new_img_file = self.img_root / "/".join(meta["img_file"].split("/")[-3:])
        img_resized = cv2.imread(str(new_img_file))[..., [2, 1, 0]]  # (H, W, 3), in RGB, uint8
        img, bbx_lurb_resized_new, A = squared_crop_and_resize(self, img_resized, bbx_lurb_resized)
        bbx_lurb = bbx_lurb_resized_new * self.PROJ_SCALE
        img = img.transpose(2, 0, 1) / 255.0  # (3, H, W), float

        data.update(
            {
                "bbx_lurb_human": torch.from_numpy(bbx_lurb_human),  # (4,)
                "bbx_lurb": torch.from_numpy(bbx_lurb).float(),  # (4,)
                "image": torch.from_numpy(img).float(),  # (rgb, H, W)
            }
        )

        return data

    def _idx2meta(self):
        idx2meta_file = self.meta_root / "idx2meta.pkl"
        return pickle.load(open(str(idx2meta_file), "rb"))

    def _img2gtbbx(self):
        bbx_file = self.meta_root / "img2gtbbx.pkl"
        return pickle.load(open(str(bbx_file), "rb"))
