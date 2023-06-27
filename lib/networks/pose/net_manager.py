from torch import nn
from lib.utils import logger
from lib.utils.net_utils import instantiate


class Network(nn.Module):
    def __init__(self, network_name, **kwargs):
        super().__init__()
        self.network_name = network_name
        logger.info(f"Instantiate the network: {network_name}")
        self.net = instantiate(kwargs[network_name])

    def forward(self, batch):
        self.net(batch)
