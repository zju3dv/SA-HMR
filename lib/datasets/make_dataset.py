from lib.utils import logger
import torch
import torch.utils.data
from time import time
from torch.utils.data.dataloader import default_collate
from lib.utils.net_utils import instantiate


def make_dataset(cfg, split="train"):
    tic = time()
    dat_cfg = cfg.get(f"{split}_dat")
    logger.info(f"Dataset {split}: {dat_cfg.target}")

    dataset = instantiate(dat_cfg)
    limit_size = dat_cfg.limit_size
    if limit_size > 0 and len(dataset) > limit_size:
        logger.warning(f"Working on subset of size {limit_size}")
        dataset = torch.utils.data.Subset(dataset, list(range(limit_size)))

    logger.debug(f"Time for making dataset: {time() - tic:.2f}s")
    return dataset


def collate_fn_wrapper(batch):
    keys_to_collate_as_list = ["meta"]
    list_in_batch = {}
    for k in keys_to_collate_as_list:
        if k in batch[0]:
            list_in_batch[k] = [data[k] for data in batch]
    # use default collate for the rest of batch
    batch = default_collate(batch)
    batch.update({k: v for k, v in list_in_batch.items()})
    return batch


def make_data_sampler(dataset, shuffle, is_distributed):
    if is_distributed:
        return torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)
    else:
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    return sampler


def make_data_loader(cfg, split="train"):
    dataset = make_dataset(cfg.dataset, split)
    logger.info(f"Final {split} dataset size: {len(dataset)}")

    datloader_cfg = cfg.get(split)
    batch_size = datloader_cfg.batch_size
    num_workers = datloader_cfg.num_workers
    shuffle = datloader_cfg.shuffle

    sampler = make_data_sampler(dataset, shuffle, is_distributed=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=split == "train" and num_workers > 0,
        collate_fn=collate_fn_wrapper,
    )

    return dataloader
