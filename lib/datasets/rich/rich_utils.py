import torch
import torch.nn as nn
import cv2
import numpy as np
from lib.utils.geo_transform import apply_T_on_points, project_p2d
from smplx.lbs import transform_mat


def sample_idx2meta(dataset, idx2meta):
    """
    1. remove frames that < 45
    2. sample frames by sample_interval
    3. sorted
    """
    idx2meta = [
        v
        for k, v in idx2meta.items()
        if int(v["frame_name"]) > 45 and (int(v["frame_name"]) + int(v["cam_id"])) % dataset.sample_interval == 0
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


def get_bbx(dataset, data):
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


def extract_cam_param_xml(xml_path="", dtype=torch.float32):
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)

    extrinsics_mat = [float(s) for s in tree.find("./CameraMatrix/data").text.split()]
    intrinsics_mat = [float(s) for s in tree.find("./Intrinsics/data").text.split()]
    distortion_vec = [float(s) for s in tree.find("./Distortion/data").text.split()]

    focal_length_x = intrinsics_mat[0]
    focal_length_y = intrinsics_mat[4]
    center = torch.tensor([[intrinsics_mat[2], intrinsics_mat[5]]], dtype=dtype)

    rotation = torch.tensor(
        [
            [extrinsics_mat[0], extrinsics_mat[1], extrinsics_mat[2]],
            [extrinsics_mat[4], extrinsics_mat[5], extrinsics_mat[6]],
            [extrinsics_mat[8], extrinsics_mat[9], extrinsics_mat[10]],
        ],
        dtype=dtype,
    )

    translation = torch.tensor([[extrinsics_mat[3], extrinsics_mat[7], extrinsics_mat[11]]], dtype=dtype)

    # t = -Rc --> c = -R^Tt
    cam_center = [
        -extrinsics_mat[0] * extrinsics_mat[3]
        - extrinsics_mat[4] * extrinsics_mat[7]
        - extrinsics_mat[8] * extrinsics_mat[11],
        -extrinsics_mat[1] * extrinsics_mat[3]
        - extrinsics_mat[5] * extrinsics_mat[7]
        - extrinsics_mat[9] * extrinsics_mat[11],
        -extrinsics_mat[2] * extrinsics_mat[3]
        - extrinsics_mat[6] * extrinsics_mat[7]
        - extrinsics_mat[10] * extrinsics_mat[11],
    ]

    cam_center = torch.tensor([cam_center], dtype=dtype)

    k1 = torch.tensor([distortion_vec[0]], dtype=dtype)
    k2 = torch.tensor([distortion_vec[1]], dtype=dtype)

    return focal_length_x, focal_length_y, center, rotation, translation, cam_center, k1, k2


class CalibratedCamera(nn.Module):
    def __init__(
        self,
        calib_path="",
        rotation=None,
        translation=None,
        focal_length_x=None,
        focal_length_y=None,
        batch_size=1,
        center=None,
        dtype=torch.float32,
        **kwargs
    ):
        super(CalibratedCamera, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        self.calib_path = calib_path
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer("zero", torch.zeros([batch_size], dtype=dtype))

        import os.path as osp

        if not osp.exists(calib_path):
            raise FileNotFoundError("Could" "t find {}.".format(calib_path))
        else:
            focal_length_x, focal_length_y, center, rotation, translation, cam_center, _, _ = extract_cam_param_xml(
                xml_path=calib_path, dtype=dtype
            )

        if focal_length_x is None or type(focal_length_x) == float:
            focal_length_x = torch.full([batch_size], focal_length_x, dtype=dtype)

        if focal_length_y is None or type(focal_length_y) == float:
            focal_length_y = torch.full([batch_size], focal_length_y, dtype=dtype)

        self.register_buffer("focal_length_x", focal_length_x)
        self.register_buffer("focal_length_y", focal_length_y)

        if center is None:
            center = torch.zeros([batch_size, 2], dtype=dtype)
        self.register_buffer("center", center)

        rotation = rotation.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        rotation = nn.Parameter(rotation, requires_grad=False)

        self.register_parameter("rotation", rotation)

        if translation is None:
            translation = torch.zeros([batch_size, 3], dtype=dtype)

        translation = translation.view(3, -1).repeat(batch_size, 1, 1).squeeze(dim=-1)
        translation = nn.Parameter(translation, requires_grad=False)
        self.register_parameter("translation", translation)

        cam_center = nn.Parameter(cam_center, requires_grad=False)
        self.register_parameter("cam_center", cam_center)

    def forward(self, points):
        device = points.device

        with torch.no_grad():
            camera_mat = torch.zeros([self.batch_size, 2, 2], dtype=self.dtype, device=points.device)
            camera_mat[:, 0, 0] = self.focal_length_x
            camera_mat[:, 1, 1] = self.focal_length_y

        camera_transform = transform_mat(self.rotation, self.translation.unsqueeze(dim=-1))
        homog_coord = torch.ones(list(points.shape)[:-1] + [1], dtype=points.dtype, device=device)
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)

        projected_points = torch.einsum("bki,bji->bjk", [camera_transform, points_h])

        img_points = torch.div(projected_points[:, :, :2], projected_points[:, :, 2].unsqueeze(dim=-1))
        img_points = torch.einsum("bki,bji->bjk", [camera_mat, img_points]) + self.center.unsqueeze(dim=1)
        return img_points
