from collections import Counter
from lib.utils.optimizer.lr_scheduler import WarmupMultiStepLR, MultiStepLR, ExponentialLR


def make_lr_scheduler(cfg_scheduler, optimizer):
    if cfg_scheduler.type == "multi_step":
        scheduler = MultiStepLR(
            optimizer,
            milestones=cfg_scheduler.milestones,
            gamma=cfg_scheduler.gamma,
        )
        scheduler.milestones = Counter(cfg_scheduler.milestones)
        scheduler.gamma = cfg_scheduler.gamma
    elif cfg_scheduler.type == "exponential":
        scheduler = ExponentialLR(
            optimizer,
            decay_epochs=cfg_scheduler.decay_epochs,
            gamma=cfg_scheduler.gamma,
        )
        scheduler.decay_epochs = cfg_scheduler.decay_epochs
        scheduler.gamma = cfg_scheduler.gamma
    else:
        raise ValueError(f"Unknown scheduler type: {cfg_scheduler.type}")

    return scheduler
