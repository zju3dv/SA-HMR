import cv2
import torch
import trimesh
import numpy as np
from torchvision.utils import make_grid


# ========== Visualize Methods ========== #

def visualize_metro(trainer, batch):
    image = batch['image']
    pred_cr_verts = batch['pred_cr_verts']
    pred_cam = batch['pred_cam']

    images = image.cpu().numpy().transpose(0, 2, 3, 1)  # (B, H, W, 3)
    cameras = pred_cam.detach().cpu().numpy()  # (B, 3)
    meshes = [trimesh.Trimesh(v, trainer.smpl_face) for v in pred_cr_verts.detach().cpu().numpy()]
    rend_images = batch['mesh_renderer'].render_ortho(meshes, cameras, images)
    # cv2.imwrite('metro_pred.jpg', np.asarray(rend_image[..., [2, 1, 0]]))
    return rend_images

def visualize_bstro(trainer, batch):
    img = batch['image'][0].cpu()
    pred_contact = batch['pred_contact'][0].detach().cpu()
    smpl = batch['smpl'].cpu()

    import cv2
    pred_contact_meshes = visualize_contact(pred_contact, smpl)
    cv2.imwrite('bstro_input.jpg', np.asarray(img.permute(1, 2, 0)[..., [2, 1, 0]] * 255))
    pred_contact_meshes.export('bstro_pred.obj')


# ===== Utils ===== #

def visualize_contact(pred_contact, smpl):
    """
    image: (3, H, W)
    pred_contact: (6890, 1)
    smpl: smpl
    """
    hit_id = (pred_contact >= 0.5).nonzero()[:, 0]
    ref_vert = smpl(torch.zeros((1, 72)), torch.zeros((1, 10))).squeeze()
    pred_mesh = trimesh.Trimesh(vertices=ref_vert.detach().numpy(), faces=smpl.faces.detach().numpy(), process=False)
    pred_mesh.visual.vertex_colors = (191, 191, 191, 255)
    pred_mesh.visual.vertex_colors[hit_id, :] = (255, 0, 0, 255)
    return pred_mesh


name2vis = {
    'visualize_metro': visualize_metro,
    'visualize_bstro': visualize_bstro,
}
