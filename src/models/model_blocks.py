import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
        Regular Convolution + BatchNorm + ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      stride=stride,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x), 'Conv_Block'


class ChoiceBlock(nn.Module):
    """
        Block with different structural options
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = ConvBlock(in_channels, out_channels,
                               kernel_size=3,
                               padding=1,
                               stride=1)

    def forward(self, x):
        return self.block(x), [-1, 'ConvBlock', [
                                                    self.in_channels,
                                                    self.out_channels,
                                                    3, 1, 1]
                               ]


class Cell(nn.Module):
    """
        Group of Choice Blocks
    """

    def __init__(self, cell_nbr, n_choice_blocks, in_channels):
        super().__init__()

        self.cell_nbr = cell_nbr
        self.n_choice_blocks = n_choice_blocks
        self.modules_dict = nn.ModuleDict(
            {f'{self.cell_nbr}_choice_block_{i}': ChoiceBlock(in_channels, in_channels)
                for i in range(n_choice_blocks)}
        )

    def forward(self, x, cell_config):
        cell_name = f'cell_{self.cell_nbr}'
        used_architecture = {cell_name: []}

        assert cell_config >= 1
        for i in range(cell_config):
            choice_block_name = f'{self.cell_nbr}_choice_block_{i}'
            (x, d_b_desc), block_desc = self.modules_dict[choice_block_name](x)
            used_architecture[cell_name].append((choice_block_name, block_desc))

        return x, used_architecture
