from lib.utils import logger
from .pose.evaluator import PoseEvaluator


def make_evaluator(cfg):
    if cfg.skip_eval or len(cfg.eval_metrics) == 0:
        logger.info("Skipping evaluation")
        return None
    else:
        logger.info("Making evaluator pose.evaluator.PoseEvaluator")
        return PoseEvaluator(cfg.eval_metrics)
