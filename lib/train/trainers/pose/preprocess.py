from lib.utils import logger
from .supervision import *
from lib.utils.geo_transform import apply_T_on_points, unproj_bbx_to_fst


def get_pc_in_frustum(trainer, batch, z_near=0.5, z_far=12.5):
    B = batch["image"].size(0)

    # Get frustum corners in the world
    c_frustum_points = unproj_bbx_to_fst(batch["bbx_lurb"], batch["K"], z_near, z_far)
    # batch['vis_c_fst_lines'] = get_lines_of_my_frustum(c_frustum_points)  # for vis-only, 0.1ms
    batch["T_c2w"] = batch["T_w2c"].inverse()
    w_frustum_points = apply_T_on_points(c_frustum_points, batch["T_c2w"])

    # Find points that fall in the frustum
    w_pcFst_all = []
    for b in range(B):
        pc_w = trainer.__getattr__(f"{batch['meta'][b]['scene_key']}_voxel")
        # when a point falls on the outer side of the face: (point-origin).dot(vec_outer) > 0
        xo = w_frustum_points[b, [0, 0, 1, 2, 1, 4]]  # (F, 3)
        xa = w_frustum_points[b, [4, 1, 2, 3, 0, 5]]
        xb = w_frustum_points[b, [3, 4, 5, 6, 2, 7]]
        mask = (torch.einsum("nfc,fc->nf", pc_w[:, None] - xo, (xa - xo).cross(xb - xo)) <= 0).all(-1)
        w_pcFst_all.append(pc_w[mask])

    c_pcFst_all = [apply_T_on_points(p[None], batch["T_w2c"][[b]])[0] for b, p in enumerate(w_pcFst_all)]
    batch["w_pcFst_all"] = w_pcFst_all
    batch["c_pcFst_all"] = c_pcFst_all
