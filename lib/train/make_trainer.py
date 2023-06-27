from .trainer import Trainer
from lib.utils.net_utils import instantiate


def make_trainer(cfg, network, base_trainer=Trainer):
    network = instantiate(cfg.trainer, net=network)
    return base_trainer(network, cfg)
