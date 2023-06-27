import os
import imp
from lib.utils import logger


def make_network(net_cfg, cfg=None):
    """
    args:
        net_cfg: local config to construct the network
        cfg: gloabl config, for rapid reseach
    """

    name = net_cfg.name
    logger.info("Making network: {}".format(name))
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    module = tmp.replace("/", ".")
    path = tmp + ".py"
    network = imp.load_source(module, path).Network(**net_cfg.get("args", {}), cfg=cfg)
    return network
