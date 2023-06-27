import os
from omegaconf import OmegaConf
from contextlib import redirect_stdout
from lib.utils import logger, logger_activate_debug_mode


# ===== Default Configs ===== #
_cfg = OmegaConf.create(
    {
        "task": "",
        "record_dir": "auto",
        "model_dir": "auto",
        "debug": False,
        "is_train": True,
        "gpus": [0],
        "resume": False,
    }
)


# ===== Utils ===== #
def make_cfg(args, skip_mkdir=False):
    """args: requires 'cfg_file' key by dot access"""

    # 1. Use cfg_file to overwrite default cfg
    logger.info(f"Making config: {args.cfg_file}")
    default_cfg_path = f"configs/{args.cfg_file.split('/')[1]}/default.yaml"
    cfg = _cfg.copy()
    if os.path.exists(default_cfg_path):
        logger.info(f"default_cfg_path: {default_cfg_path}")
        cfg.default_cfg_path = default_cfg_path  # record default config path
        cfg = OmegaConf.merge(cfg, OmegaConf.load(default_cfg_path))
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.cfg_file))
    else:
        default_cfg_path = OmegaConf.load(args.cfg_file)["default_cfg_path"]
        logger.info(f"default_cfg_path: {default_cfg_path}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(default_cfg_path))
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.cfg_file))
        cfg.gpus = [0]
        cfg.resume = False

    # 2. Load args.opts
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(getattr(args, "opts", [])))

    # 3. Setup environment
    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(gpu) for gpu in cfg.gpus])
    os.environ["EGL_DEVICE_ID"] = str(min(cfg.gpus))
    cfg.is_train = not getattr(args, "is_test", False)
    cfg.debug = getattr(args, "debug", False)

    # 4. Setup tb and weights dir
    if len(cfg.task) > 0 and skip_mkdir == False:
        if cfg.record_dir == "auto":
            cfg.record_dir = f"out/train/{cfg.task}"
        os.system(f"mkdir -p {cfg.record_dir}")

        if cfg.model_dir == "auto":
            cfg.model_dir = f"out/train/{cfg.task}/model"
            os.system(f"mkdir -p {cfg.model_dir}")

    # 5. logging with debug mode
    if cfg.debug:
        logger_activate_debug_mode()
        logger.debug(cfg)

    return cfg


def save_cfg(cfg, resmue=False):
    filename = f"{cfg.record_dir}/config.yaml"
    if resmue:
        filename += ".resume"
    with open(filename, "w") as f:
        logger.info(f"Saving cfg into {filename}")
        with redirect_stdout(f):
            print(cfg)
