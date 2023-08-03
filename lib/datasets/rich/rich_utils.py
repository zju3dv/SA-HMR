import torch
import cv2
import numpy as np
from lib.utils.geo_transform import apply_T_on_points, project_p2d
from pathlib import Path


# ----- Meta sample utils ----- #


def sample_idx2meta(idx2meta, sample_interval):
    """
    1. remove frames that < 45
    2. sample frames by sample_interval
    3. sorted
    """
    idx2meta = [
        v
        for k, v in idx2meta.items()
        if int(v["frame_name"]) > 45 and (int(v["frame_name"]) + int(v["cam_id"])) % sample_interval == 0
    ]
    idx2meta = sorted(idx2meta, key=lambda meta: meta["img_key"])
    return idx2meta


def remove_bbx_invisible_frame(idx2meta, img2gtbbx):
    raw_img_lu = np.array([0.0, 0.0])
    raw_img_rb_type1 = np.array([4112.0, 3008.0]) - 1  # horizontal
    raw_img_rb_type2 = np.array([3008.0, 4112.0]) - 1  # vertical

    idx2meta_new = []
    for meta in idx2meta:
        gtbbx_center = np.array([img2gtbbx[meta["img_key"]][[0, 2]].mean(), img2gtbbx[meta["img_key"]][[1, 3]].mean()])
        if (gtbbx_center < raw_img_lu).any():
            continue
        raw_img_rb = raw_img_rb_type1 if meta["cam_key"] not in ["Pavallion_3", "Pavallion_5"] else raw_img_rb_type2
        if (gtbbx_center > raw_img_rb).any():
            continue
        idx2meta_new.append(meta)
    return idx2meta_new


def remove_extra_rules(idx2meta):
    multi_person_seqs = ["LectureHall_009_021_reparingprojector1"]
    idx2meta = [meta for meta in idx2meta if meta["seq_name"] not in multi_person_seqs]
    return idx2meta


# ----- Image utils ----- #


def compute_bbx(dataset, data):
    """
    Use gt_smplh_params to compute bbx (w.r.t. original image resolution)
    Args:
        dataset: rich_pose.RichPose
        data: dict

    # This function need extra scripts to run
    from lib.utils.smplx_utils import make_smplx
    self.smplh_male = make_smplx("rich-smplh", gender="male")
    self.smplh_female = make_smplx("rich-smplh", gender="female")
    self.smplh = {
        "male": self.smplh_male,
        "female": self.smplh_female,
    }
    """
    gender = data["meta"]["gender"]
    smplh_params = {k: v.reshape(1, -1) for k, v in data["gt_smplh_params"].items()}
    smplh_opt = dataset.smplh[gender](**smplh_params)
    verts_3d_w = smplh_opt.vertices
    T_w2c, K = data["T_w2c"], data["K"]
    verts_3d_c = apply_T_on_points(verts_3d_w, T_w2c[None])
    verts_2d = project_p2d(verts_3d_c, K[None])[0]
    min_2d = verts_2d.T.min(-1)[0]
    max_2d = verts_2d.T.max(-1)[0]
    bbx = torch.stack([min_2d, max_2d]).reshape(-1).numpy()
    return bbx


def get_2d(dataset, data):
    gender = data["meta"]["gender"]
    smplh_params = {k: v.reshape(1, -1) for k, v in data["gt_smplh_params"].items()}
    smplh_opt = dataset.smplh[gender](**smplh_params)
    joints_3d_w = smplh_opt.joints
    T_w2c, K = data["T_w2c"], data["K"]
    joints_3d_c = apply_T_on_points(joints_3d_w, T_w2c[None])
    joints_2d = project_p2d(joints_3d_c, K[None])[0]
    conf = torch.ones((73, 1))
    keypoints = torch.cat([joints_2d, conf], dim=1)
    return keypoints


def squared_crop_and_resize(dataset, img, bbx_lurb, dst_size=224):
    center_rand = dataset.BBX_CENTER * (np.random.random(2) * 2 - 1)
    center_x = (bbx_lurb[0] + bbx_lurb[2]) / 2 + center_rand[0]
    center_y = (bbx_lurb[1] + bbx_lurb[3]) / 2 + center_rand[1]
    ori_half_size = max(bbx_lurb[2] - bbx_lurb[0], bbx_lurb[3] - bbx_lurb[1]) / 2
    ori_half_size *= 1 + 0.15 + dataset.BBX_ZOOM * np.random.random()  # zoom

    src = np.array(
        [
            [center_x - ori_half_size, center_y - ori_half_size],
            [center_x + ori_half_size, center_y - ori_half_size],
            [center_x, center_y],
        ],
        dtype=np.float32,
    )
    dst = np.array([[0, 0], [dst_size - 1, 0], [dst_size / 2 - 0.5, dst_size / 2 - 0.5]], dtype=np.float32)

    A = cv2.getAffineTransform(src, dst)
    img_crop = cv2.warpAffine(img, A, (dst_size, dst_size), flags=cv2.INTER_LINEAR)
    bbx_new = np.array(
        [center_x - ori_half_size, center_y - ori_half_size, center_x + ori_half_size, center_y + ori_half_size],
        dtype=bbx_lurb.dtype,
    )
    return img_crop, bbx_new, A


# ----- Camera utils ----- #


def extract_cam_xml(xml_path="", dtype=torch.float32):
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)

    extrinsics_mat = [float(s) for s in tree.find("./CameraMatrix/data").text.split()]
    intrinsics_mat = [float(s) for s in tree.find("./Intrinsics/data").text.split()]
    distortion_vec = [float(s) for s in tree.find("./Distortion/data").text.split()]

    return {
        "ext_mat": torch.tensor(extrinsics_mat).float(),
        "int_mat": torch.tensor(intrinsics_mat).float(),
        "dis_vec": torch.tensor(distortion_vec).float(),
    }


def get_cam2params(scene_info_root):
    """
    Args:
        scene_info_root: this could be repalced by path to scan_calibration
    """
    cam_params = {}
    cam_xml_files = Path(scene_info_root).glob("*/calibration/*.xml")
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
