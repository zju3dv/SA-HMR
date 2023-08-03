import argparse
import pickle
from tqdm import tqdm
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from lib.config import make_cfg
from lib.utils import logger
from lib.train.param_utils import load_pretrained_network
from lib.utils.net_utils import to_cuda, instantiate
from lib.datasets.rich.rich_pose_test import Dataset as RICH_Dataset
from lib.datasets.prox.prox_pose_quant import Dataset as PROX_quant_Dataset
from lib.datasets.make_dataset import collate_fn_wrapper


def main(cfg):
    data_name = cfg.data_name
    ckpt_fn = cfg.pretrained_ckpt

    network = instantiate(cfg.network)
    load_pretrained_network(ckpt_fn, network)
    network = instantiate(cfg.network_wrapper, net=network).eval().cuda()  # wrapper with inference utils

    # data
    if data_name == "rich":
        dataset = RICH_Dataset()
        data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_wrapper)
    elif data_name == "prox_quant":
        dataset = PROX_quant_Dataset()
        data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_wrapper)

    # output dir
    out_dir = Path(f"out/pred_dump/{cfg.task}")
    out_dir.mkdir(exist_ok=True, parents=True)

    for idx, batch in enumerate(tqdm(data_loader)):
        # assume batch-size == 1
        meta = batch["meta"][0]
        img_key = meta["img_key"]
        out_fn = out_dir / f"{img_key}.pkl"

        # run
        batch = to_cuda(batch)
        with torch.no_grad():
            network(batch, compute_supervision=False, compute_loss=False)

        # dump output
        pred_c_verts = batch["pred_c_verts"]
        pred_c_pelvis = batch["pred_c_pelvis"]
        pred_c_pelvis_refined = batch["pred_c_pelvis_refined"]
        save_dict = {
            "pred_c_verts": pred_c_verts.detach().cpu().numpy()[0],
            "pred_c_pelvis": pred_c_pelvis.detach().cpu().numpy()[0],
            "pred_c_pelvis_refined": pred_c_pelvis_refined.detach().cpu().numpy()[0],
        }

        with open(out_fn, "wb") as f:
            pickle.dump(save_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", "-c", type=str, required=True)

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = make_cfg(args)

    main(cfg)
