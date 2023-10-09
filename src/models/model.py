import yaml
import torch
import torch.nn as nn

from src.models.model_blocks import *
from src.utils.utils import load_config, init_weights_kaiming_normal


def load_model(path_model_config, path_weights=None, device='cpu'):
    """
    Creates a model based on a config file and loads weights if provided

    :param path_model_config: path to model's config yaml file
    :param path_weights: path to pretrained weights
    :param device: device type
    :return: model - supernet or sampled from it
    """

    model = Supernet(path_model_config)
    model.to(device)
    # model.apply(init_weights_kaiming_normal)

    if path_weights:
        loaded_weights = torch.load(path_weights)
        loaded_layer_names = list(loaded_weights.keys())

        sd = model.state_dict()
        with torch.no_grad():
            for layer_name in sd:
                if layer_name in loaded_layer_names:
                    sd[layer_name].data = loaded_weights[layer_name].data
        model.load_state_dict(sd)

    return model


def sample_from_supernet(path_supernet_config, sample_arch_config,
                         path_supernet_weights=None):
    """
    Samples stand-alone model from supernet
    """

    if isinstance(sample_arch_config, list):
        supernet_config = load_config(path_supernet_config)

        sample_config = {'nbr_cells': len(sample_arch_config),
                         'nbr_choice_blocks': sample_arch_config,
                         'backbone': [],
                         'nc': 10,
                         'img_W': 32,
                         'img_H': 32,
                         'channels': 3,
                         'lr': 1e-3,
                         'momentum': 0.9,
                         'decay': 0.0005,
                         'batch_size': 32
                         }
        # TODO fix the mess above
        for layer in supernet_config['backbone']:
            if 'Cell' in layer[1]:
                layer[2][1] = sample_arch_config[layer[2][0] - 1]
                sample_config['backbone'].append(layer)
            else:
                sample_config['backbone'].append(layer)

        sampled_yaml = f'{path_supernet_config.replace(".yaml","")}_{sample_arch_config}.yaml'
        with open(sampled_yaml, 'w') as f:
            doc = yaml.dump(sample_config, f)

        if path_supernet_weights:
            model = load_model(sampled_yaml, path_weights=path_supernet_weights)

            sampled_pt = f'{path_supernet_config.replace("yaml", "")}_{sample_arch_config}.pt'
            torch.save(model.state_dict(), sampled_pt)
        else:
            model = Supernet(sampled_yaml)
            sampled_pt = None

        return model, sampled_pt, sampled_yaml

    else:
        # TODO implement sampling from used modules list
        #  from cfg file - use load_model
        raise NotImplementedError('Only list sample method is covered')


def create_model(config):
    """
    Creates model (Module dict) from yaml config file

    :param config: loaded config yaml file
    :return: hyperparams, modules_dict
    """

    hyperparams = {
        'nbr_cells': int(config['nbr_cells']),
        'nbr_choice_blocks': config['nbr_choice_blocks'],
        'nc': int(config['nc']),
        'img_W': int(config['img_W']),
        'img_H': int(config['img_H']),
        'n_channels': int(config['channels']),
        'lr': float(config['lr']),
        'momentum': float(config['momentum']),
        'decay': float(config['decay']),
        'batch_size': int(config['batch_size']),
    }
    assert hyperparams['img_H'] == hyperparams['img_W'] == 32

    modules_dict = nn.ModuleDict()
    for i, layer_info in enumerate(config['backbone']):
        if 'ConvBlock' in layer_info[1]:
            modules_dict.update({
                layer_info[1]: ConvBlock(
                                in_channels=layer_info[2][0],
                                out_channels=layer_info[2][1],
                                kernel_size=layer_info[2][2],
                                padding=layer_info[2][3],
                                stride=layer_info[2][4],
                                )
            })
        elif 'Cell' in layer_info[1]:
            modules_dict.update({
                layer_info[1]: Cell(
                                cell_nbr=layer_info[2][0],
                                n_choice_blocks=layer_info[2][1],
                                in_channels=layer_info[2][2],
                                )
            })
        elif layer_info[1] == 'AvgPool':
            modules_dict.update({
                'AvgPool': nn.AvgPool2d(
                                kernel_size=layer_info[2][0],
                                padding=layer_info[2][1],
                                stride=layer_info[2][2],
                                )
            })
        elif layer_info[1] == 'Flatten':
            modules_dict.update({'Flatten': nn.Flatten()})
        elif layer_info[1] == 'Linear':
            modules_dict.update({
                'Linear': nn.Linear(
                        in_features=layer_info[2][0],
                        out_features=layer_info[2][1]
                        )
            })

    return hyperparams, modules_dict


class Supernet(nn.Module):
    """
        Supernet model with Choice blocks
    """

    def __init__(self, config):
        super().__init__()
        if isinstance(config, str):
            self.config = load_config(config)
        else:
            self.config = config
        self.hyperparams, self.modules_dict = create_model(self.config)

    def forward(self, x, cells_config):
        cell_cnt = 0
        architecture_desc_list = []
        for module_i, (module_info, (module_name, module)) in enumerate(zip(self.config['backbone'], self.modules_dict.items())):
            if 'Cell' in module_name:
                x, architecture_desc = module(x, cells_config[cell_cnt])
                # For now we need only number of used choice_blocks
                #
                architecture_desc_list.append([module_name,
                                               len(list(architecture_desc.values())[0])])
                cell_cnt += 1
            elif 'ConvBlock' in module_name:
                x, architecture_desc = module(x)
                architecture_desc_list.append(architecture_desc)
            elif module_name in ['Flatten', 'Linear', 'AvgPool']:
                x = module(x)
                architecture_desc_list.append(module_name)

        return x, architecture_desc_list

