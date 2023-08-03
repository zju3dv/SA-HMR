import torch
from lib.utils import logger


def load_pretrained_network(ckpt_fn, network: torch.nn.Module):
    """
    Load ckpt to network
    """
    if ckpt_fn is None:
        logger.info("[Loading Pretrained (all)] No ckpt_fn provided, skip loading")
        return

    logger.info(f"Load path: {ckpt_fn}")
    ckpt = torch.load(ckpt_fn, "cpu")["network"]
    remove_prefix = "net."
    try:
        # To accommodate the released ckpt
        ckpt = {k[len(remove_prefix) :]: v for k, v in ckpt.items()}
        missing_keys, unexpected_keys = network.load_state_dict(ckpt, strict=False)
        if len(missing_keys) > 0:
            logger.info(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.info(f"Unexpected keys: {unexpected_keys}")
    except RuntimeError as e:
        params_names = [s.split(":")[0] for s in e.args[0].split("mismatch for ")[1:]]
        logger.info(f"pretrained size mismatch: {params_names}")
        for k, v in ckpt.items():  # sanity check
            if k not in params_names:
                assert ~(v != dict(network.named_parameters())[k]).sum()


def select_trainable_params(finetune_strategy, network: torch.nn.Module):
    # print the parameters that are trainable
    if finetune_strategy is None:
        logger.info("All parameters are trainable")
        return network.parameters()
    elif finetune_strategy == "metro_enhanced_only":
        logger.info("Only metro_enhanced is trainable")
        return network.net.metro_enhanced.parameters()
    else:
        raise ValueError(f"Unknown finetune_strategy: {finetune_strategy}")
