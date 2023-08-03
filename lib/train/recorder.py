from collections import deque, defaultdict
import torch
from tensorboardX import SummaryWriter
from lib.utils import logger
from pathlib import Path


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class Recorder(object):
    def __init__(self, exp_dir):
        logdir = Path(exp_dir) / "tensorboard"
        self.writer = SummaryWriter(logdir=logdir)
        logger.info(f"Record at {logdir}")

        # scalars
        self.epoch = 0
        self.step = 0
        self.loss_stats = defaultdict(SmoothedValue)
        self.batch_time = SmoothedValue()
        self.data_time = SmoothedValue()

    def update_loss_stats(self, loss_dict):
        for k, v in loss_dict.items():
            self.loss_stats[k].update(v.detach().cpu())

    def record(self, prefix, step=-1, loss_stats=None, image_stats=None):
        pattern = prefix + "/{}"
        step = step if step >= 0 else self.step
        loss_stats = loss_stats if loss_stats else self.loss_stats

        for k, v in loss_stats.items():
            if isinstance(v, SmoothedValue):
                self.writer.add_scalar(pattern.format(k), v.median, step)
            else:
                self.writer.add_scalar(pattern.format(k), v, step)

    def state_dict(self):
        scalar_dict = {}
        scalar_dict["step"] = self.step
        return scalar_dict

    def load_state_dict(self, scalar_dict):
        self.step = scalar_dict["step"]

    def __str__(self):
        loss_state = []
        for k, v in self.loss_stats.items():
            loss_state.append("{}: {:.4f}".format(k, v.avg))
        loss_state = "  ".join(loss_state)

        recording_state = "  ".join(["epoch: {}", "step: {}", "{}", "data: {:.4f}", "batch: {:.4f}"])
        return recording_state.format(self.epoch, self.step, loss_state, self.data_time.avg, self.batch_time.avg)


def make_recorder(exp_dir):
    return Recorder(exp_dir)
