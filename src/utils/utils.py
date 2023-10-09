import random

import yaml
import itertools
import torch.nn as nn


def load_config(path_config):
    with open(path_config) as f:
        config = yaml.safe_load(f)
    return config


def init_weights_kaiming_normal(m):
    class_name = m.__class__.__name__
    print(class_name)
    print(m.weight.shape)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.01)
    elif class_name.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight)
    # elif isinstance(m, nn.BatchNorm2d):
    #     nn.init.kaiming_normal_(m.weight)
    #     m.bias.data.fill(0.01)


def all_cell_configs(nbr_cells, nbr_choice_blocks):
    return list(itertools.product(*[list(range(1, nbr_choice_blocks + 1))
                                  for i in range(nbr_cells)]))


def random_cell_config(nbr_cells, nbr_choice_blocks):
    """"
        Function that returns a list of configs for each cell
    """
# TODO improve distribution to balance architectures
    configs = list(itertools.product(*[list(range(1, nbr_choice_blocks + 1))
                                       for i in range(nbr_cells)]))
    return configs[random.randint(0, nbr_choice_blocks**nbr_cells - 1)]
