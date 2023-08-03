from omegaconf import OmegaConf
from lib.utils import logger, logger_activate_debug_mode


def make_cfg(args):
    """A minimal implemented hydra-like config making function"""
    default_cfg_path = "configs/pose/default.yaml"
    exp_cfg_path = args.cfg_file

    # 1. Load default config
    logger.info(f"Config: default= {default_cfg_path}")
    cfg = OmegaConf.load(default_cfg_path)

    # 2. Use cfg_file to overwrite the file pointers in defaults,
    #    then load the file, and overwrite the cfg
    exp_cfg = OmegaConf.load(exp_cfg_path)
    if "defaults" in exp_cfg:
        cfg.defaults = OmegaConf.merge(cfg.defaults, exp_cfg.defaults)
        for k, v in cfg.defaults.items():
            if v is not None:  # Load the file, and overwrite cfg
                assert k not in cfg
                cfg[k] = OmegaConf.load(v)

    # 3. Use exp_cfg to overwrite cfg
    #    by doing step2 and step3, we can overwrite defaults by exp_cfg
    logger.info(f"Config: cfg_file= {exp_cfg_path}")
    cfg = OmegaConf.merge(cfg, OmegaConf.load(exp_cfg_path))

    # 4. Merge with run-time args.opts
    if getattr(args, "opts", None) is not None and len(args.opts) > 0:
        logger.info(f"Config: args.opts= {args.opts}")
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.opts))

    # Logging with debug mode
    cfg.debug = getattr(args, "debug", False)
    if cfg.debug:
        logger_activate_debug_mode()
        logger.debug(cfg)

    return cfg


def save_cfg(cfg, exp_dir):
    filename = f"{exp_dir}/config.yaml"
    # Save omegaconf config to yaml
    logger.info(f"Saving cfg into {filename}")
    OmegaConf.save(cfg, filename)
