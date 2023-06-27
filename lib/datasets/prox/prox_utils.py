import numpy as np
import cv2


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
