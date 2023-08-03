import logging
import random
import numpy as np
import torch


def seed_everything(seed=66):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


formatter = logging.Formatter(fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s", datefmt="%Y/%m/%d %H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger("SAHMR")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


def warning_on(condition, msg):
    if condition:
        logger.warning(msg)
    pass


def logger_activate_debug_mode():
    """change global logger"""
    logger.info("===== 'logger.debug()' enabled =====")
    logger.setLevel(logging.DEBUG)
    logger.handlers[0].setLevel(logging.DEBUG)
