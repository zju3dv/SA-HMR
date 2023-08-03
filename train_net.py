import argparse
from pathlib import Path
from lib.utils import seed_everything
from lib.config import make_cfg, save_cfg
from lib.utils.net_utils import save_network
from lib.train import make_recorder, Trainer, make_optimizer, make_lr_scheduler
from lib.evaluators import make_evaluator
from lib.datasets.make_dataset import make_data_loader
from lib.utils.net_utils import instantiate
from lib.train.param_utils import load_pretrained_network, select_trainable_params


def main(cfg):
    seed_everything()

    # Directory: where everything will be saved
    exp_dir = Path("out") / cfg.data_name / cfg.task
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_cfg(cfg, exp_dir)

    # Data
    data_loader = make_data_loader(cfg, split="train")
    val_loader = make_data_loader(cfg, split="val")

    # Network and its wrapper (with utilities)
    network = instantiate(cfg.network)
    load_pretrained_network(cfg.pretrained_ckpt, network)
    network = instantiate(cfg.network_wrapper, net=network)

    # Trainer: optimizer, scheduler, evaluator, recorder
    trainer = Trainer(network, cfg.log_interval, cfg.record_interval)
    net_params = select_trainable_params(cfg.finetune_strategy, network)
    optimizer = make_optimizer(cfg.train.optimizer, net_params)
    scheduler = make_lr_scheduler(cfg.train.scheduler, optimizer)
    evaluator = make_evaluator(cfg)
    recorder = make_recorder(exp_dir)

    # Training loop
    for epoch in range(1, cfg.train.epoch + 1):
        recorder.epoch = epoch

        trainer.train(epoch, data_loader, optimizer, recorder)
        scheduler.step()

        if epoch % cfg.save_ep == 0:
            save_network(network, exp_dir, epoch)

        if epoch % cfg.eval_ep == 0 and not cfg.skip_eval:
            trainer.val(epoch, val_loader, evaluator, recorder)

    if cfg.save_last:
        save_network(network, exp_dir, epoch, last=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", "-c", type=str, required=True)

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = make_cfg(args)

    main(cfg)
