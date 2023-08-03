import torch
from lib.utils import logger

_optimizer_factory = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


def make_optimizer(cfg_opt, net_params=None):
    optim = cfg_opt.optim

    # set-up learning rate
    lr = cfg_opt.lr
    bs = cfg_opt.bs
    if lr == 0:
        lr = cfg_opt.canonical_lr * (bs / cfg_opt.canonical_bs)
    logger.info(f"lr {lr:.2e}, world batchsize {bs}")

    adam_decay = cfg_opt.weight_decay
    adamw_decay = cfg_opt.adamw_weight_decay

    parameters = net_params
    if optim == "adam":
        optimizer = _optimizer_factory[optim](parameters, lr=lr, weight_decay=adam_decay)
    elif optim == "adamw":
        optimizer = _optimizer_factory[optim](parameters, lr=lr, weight_decay=adamw_decay)
    else:
        raise ValueError

    return optimizer
