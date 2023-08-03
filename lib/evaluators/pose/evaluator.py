import numpy as np
from lib.utils import logger
from lib.evaluators.pose.metrics import name2func


class PoseEvaluator:
    def __init__(self, metrics):
        self.metric_func_names = metrics  # List
        logger.info(f"Metrics Functions: {self.metric_func_names}")
        self.init_metric_stats()

    def init_metric_stats(self):
        """Call at initialization and end"""
        self.metric_stats = {}

    def update(self, k, v_list: list):
        """v_list need to be List of simple scalars"""
        if k in self.metric_stats:
            self.metric_stats[k].extend(v_list)
        else:
            self.metric_stats[k] = v_list

    def evaluate(self, batch):
        for k in self.metric_func_names:
            name2func[k](self, batch)

    def summarize(self):
        if len(self.metric_stats) == 0:
            return {}, {}

        values = [np.array(self.metric_stats[k]).flatten() for k in self.metric_stats]
        metrics_raw = {k: vs for k, vs in zip(self.metric_stats, values)}
        metrics = {k: np.mean(vs) for k, vs in zip(self.metric_stats, values)}

        message = f"Avg-over {len(values[0])}. Metrics: "
        for k, v in metrics.items():
            message += f"{k}: {v:.2f} ; "
        logger.info(message)

        self.init_metric_stats()
        return metrics, metrics_raw
