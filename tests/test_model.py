import pytest
import sys
import itertools
import torch

sys.path.append('../')

from src.models.model import create_model, sample_from_supernet, Supernet
from src.models.model_blocks import ConvBlock, ChoiceBlock, Cell
from src.utils.utils import load_config


@pytest.fixture
def load_cfg():
    cfg = load_config('../src/configs/supernet_config.yaml')
    return cfg


def test_create_model_modules_order_match(load_cfg):
    _, modules_dict = create_model(load_cfg)
    cfg_layers = [m[1] for m in load_cfg['backbone']]

    for cfg_l, modules_dict_l in zip(cfg_layers, modules_dict.keys()):
        assert cfg_l == modules_dict_l


def test_create_model_check_conv_params(load_cfg):
    _, modules_dict = create_model(load_cfg)
    cfg_layers = load_cfg['backbone']

    created_conv_in_channels = [v.conv_block[0].in_channels
                                for k, v in modules_dict.items() if 'ConvBlock' in k]
    cfg_conv_in_channels = [m[2][0] for m in cfg_layers if 'ConvBlock' in m[1]]

    created_cell_in_channels = [v.modules_dict[f'{v.cell_nbr}_choice_block_1'].in_channels
                                for k, v in modules_dict.items() if 'Cell' in k]
    cfg_cell_in_channels = [m[2][2] for m in cfg_layers if 'Cell' in m[1]]

    assert cfg_conv_in_channels == created_conv_in_channels
    assert cfg_cell_in_channels == created_cell_in_channels


def test_cell_config():
    pass


def test_conv_block_forward():
    model = ConvBlock(3, 32)
    inputs = torch.rand(16, 3, 32, 32)
    expected_output_shape = [16, 32, 32, 32]

    with torch.no_grad():
        outputs, _ = model(inputs)
    assert list(outputs.shape) == expected_output_shape


def test_choice_block_forward():
    model = ChoiceBlock(32, 32)
    inputs = torch.rand(16, 32, 124, 124)
    expected_output_shape = [16, 32, 124, 124]

    with torch.no_grad():
        (outputs, __), _ = model(inputs)
    assert list(outputs.shape) == expected_output_shape


def test_cell_forward():
    model = Cell(1, 3, 32)
    inputs = torch.rand(16, 32, 124, 124)
    expected_output_shape = [16, 32, 124, 124]

    for i in range(1, 4):
        with torch.no_grad():
            outputs, _ = model(inputs, i)
        assert list(outputs.shape) == expected_output_shape


def test_cell_backward():
    model = Cell(1, 3, 2)
    inputs = torch.rand(1, 2, 12, 12)
    parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(
        parameters,
        lr=0.5
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    target = torch.rand(1, 2, 12, 12)

    initial_weights = {}
    for name, param in model.named_parameters():
        initial_weights[name] = param.clone().detach()

    for i in range(100):
        outputs, _ = model(inputs, 2)
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    for name, param in model.named_parameters():
        if name in initial_weights:
            difference = (param - initial_weights[name]).abs().sum()
            # print(f'Layer: {name}, Weight Difference: {difference.item()}')
            if f'choice_block_0' in name or f'choice_block_1' in name:
                assert difference.item() != 0
            elif f'choice_block_2' in name:
                assert difference.item() == 0


@pytest.mark.parametrize(
    'cfg_path', ['../src/configs/supernet_config.yaml']
)
def test_forward_end_shape(cfg_path):
    model = Supernet(cfg_path)
    inputs = torch.rand(16, 3, 32, 32)
    expected_output_shape = [16, 10]
    output_shapes = []

    nbr_cells = model.config['nbr_cells']
    nbr_choice_blocks = model.config['nbr_choice_blocks']
    arch_configs = list(itertools.product(*[list(range(1, nbr_choice_blocks+1)) for i in range(nbr_cells)]))

    for arch_config in arch_configs:
        outputs, _ = model(inputs, arch_config)
        output_shapes.append(outputs.shape)

    assert all(list(output_shape) == expected_output_shape for output_shape in output_shapes)


# @pytest.mark.parametrize(
#     'cfg_path', ['../src/configs/supernet_config.yaml']
# )
# def test_load_sampled_model(cfg_path):
#     model = Supernet(cfg_path)
#     torch.save(model.state_dict(), 'supernet_dummy.pt')
#
#     inputs = torch.rand(16, 3, 32, 32)
#     outs, desc = model(inputs, [1, 3])
#
#     print(desc)
#
#     assert False


@pytest.mark.parametrize(
    'cfg_path, path_supernet_weights, sample_config',
    [('../src/configs/supernet_config.yaml', 'supernet_dummy.pt', [2, 2])]
)
def test_sample_from_supernet(cfg_path, path_supernet_weights, sample_config):
    sampled_model, path_weights, path_config = sample_from_supernet(cfg_path,
                                                                    sample_config,
                                                                    path_supernet_weights)

    choice_blocks = []
    for cell in range(1, len(sample_config) + 1):
        for block in range(sample_config[cell - 1]):
            for k in sampled_model.state_dict().keys():
                if k == f'modules_dict.Cell_{cell}.modules_dict.{cell}_choice_block_{block}.block.conv_block.1.weight':
                    choice_blocks.append(k)

    assert len(choice_blocks) == sum(sample_config)
