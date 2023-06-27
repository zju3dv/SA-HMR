import torch
import torch.nn as nn
import smplx

kwargs_disable_member_var = {
    'create_body_pose': False,
    'create_betas': False,
    'create_global_orient': False,
    'create_transl': False,
    'create_left_hand_pose': False,
    'create_right_hand_pose': False,
    'create_expression': False,
    'create_jaw_pose': False,
    'create_leye_pose': False,
    'create_reye_pose': False,
}


class BodyModelSMPLX(nn.Module):
    """Support Batch inference"""

    def __init__(self, model_path, **kwargs):
        super().__init__()
        # enable flexible batchsize, handle missing variable at forward()
        kwargs.update(kwargs_disable_member_var)
        self.bm = smplx.create(model_path=model_path, **kwargs)

        self.hand_pose_dim = self.bm.num_pca_comps if self.bm.use_pca else 3 * self.bm.NUM_HAND_JOINTS

    def forward(self,
                betas=None,
                global_orient=None,
                transl=None,
                body_pose=None,
                left_hand_pose=None,
                right_hand_pose=None,
                expression=None,
                jaw_pose=None,
                leye_pose=None,
                reye_pose=None,
                **kwargs):

        device, dtype = self.bm.shapedirs.device, self.bm.shapedirs.dtype

        model_vars = [betas, global_orient, body_pose, transl,
                      expression, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))

        if global_orient is None:
            global_orient = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if body_pose is None:
            body_pose = torch.zeros(3 * self.bm.NUM_BODY_JOINTS, device=device,
                                    dtype=dtype)[None].expand(batch_size, -1).contiguous()
        if left_hand_pose is None:
            left_hand_pose = torch.zeros(self.hand_pose_dim, device=device, dtype=dtype)[
                None].expand(batch_size, -1).contiguous()
        if right_hand_pose is None:
            right_hand_pose = torch.zeros(self.hand_pose_dim, device=device, dtype=dtype)[
                None].expand(batch_size, -1).contiguous()
        if jaw_pose is None:
            jaw_pose = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if leye_pose is None:
            leye_pose = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if reye_pose is None:
            reye_pose = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if expression is None:
            expression = torch.zeros([batch_size, self.bm.num_expression_coeffs], dtype=dtype, device=device)
        if betas is None:
            betas = torch.zeros([batch_size, self.bm.num_betas], dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        bm_out = self.bm(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=transl,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            **kwargs
        )

        return bm_out
