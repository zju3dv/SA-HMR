import torch
from lib.utils import logger


def load_pretrained_network(ckpt_fn, network: torch.nn.Module):
    """
    Load ckpt to network
    """
    logger.info(f"Load path: {ckpt_fn}")

    ckpt = torch.load(ckpt_fn, "cpu")["network"]
    try:
        missing_keys, unexpected_keys = network.load_state_dict(ckpt, strict=False)
    except RuntimeError as e:
        params_names = [s.split(":")[0] for s in e.args[0].split("mismatch for ")[1:]]
        logger.info(f"pretrained size mismatch: {params_names}")
        for k, v in ckpt.items():  # sanity check
            if k not in params_names:
                assert ~(v != dict(network.named_parameters())[k]).sum()
