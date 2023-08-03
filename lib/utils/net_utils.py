import torch
import torch.nn as nn
import os
import importlib
from collections import OrderedDict
from termcolor import colored
from lib.utils import logger
from pathlib import Path


# ----- Instantiate Utils ----- #


def instantiate(config, **kwargs):
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **kwargs)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


# ----- To Cuda ----- #


def to_cuda(batch, device="cuda"):
    for k in batch:
        if "meta" in k:
            continue
        if "mesh" in k or "name" in k:
            continue
        elif isinstance(batch[k], int):
            continue
        elif isinstance(batch[k], float):
            continue
        if isinstance(batch[k], tuple) or isinstance(batch[k], list):
            batch[k] = [b.to(device) for b in batch[k]]
        elif isinstance(batch[k], dict):
            batch[k] = {_k: _v.to(device) for _k, _v in batch[k].items()}
        else:
            batch[k] = batch[k].to(device)
    return batch


# ----- Model IO Utils ----- #


def save_network(net, exp_dir, epoch, last=False, keep=5):
    model_dir = Path(exp_dir) / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)

    # To math the format of the pretrained model    
    net_state_dict = OrderedDict()
    for k, v in net.net.state_dict().items():
        net_state_dict[f'net.{k}'] = v
    model = {"network": net_state_dict, "epoch": epoch}
    if last:
        torch.save(model, os.path.join(model_dir, "latest.pth"))
    else:
        torch.save(model, os.path.join(model_dir, "{}.pth".format(epoch)))

    # remove previous pretrained model if the number of models is too big
    pths = [int(pth.split(".")[0]) for pth in os.listdir(model_dir) if pth != "latest.pth"]
    if len(pths) <= keep:
        return
    os.system("rm {}".format(os.path.join(model_dir, "{}.pth".format(min(pths)))))


def remove_net_prefix(net, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(prefix):
            net_[k[len(prefix) :]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def add_net_prefix(net, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        net_[prefix + k] = net[k]
    return net_


def replace_net_prefix(net, orig_prefix, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(orig_prefix):
            net_[prefix + k[len(orig_prefix) :]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def remove_net_layer(net, layers):
    keys = list(net.keys())
    for k in keys:
        for layer in layers:
            if k.startswith(layer):
                del net[k]
    return net


def pick_net_layer(net, layers):
    net_ = OrderedDict()
    for k in net.keys():
        for layer in layers:
            if k.startswith(layer):
                net_[k] = net[k]
    return net_


def initialize_weights(modules, nonlinearity="relu"):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # BERT-like initialization
        # https://github.com/huggingface/transformers/blob/fcefa200b2d9636d98fd21ea3b176a09fe801c29/src/transformers/models/bert/modeling_bert.py#L745
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=0.02)
