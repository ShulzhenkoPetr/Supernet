#! /usr/bin/env python3
import argparse
import random
import numpy as np
import os
import tqdm

import torch

from src.utils.utils import all_cell_configs
from src.utils.metrics import accuracy


def evaluate(model, dataloader_val, nbr_cells, nbr_choice_blocks,
             device, passed_cfg=None, is_supernet=True):
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)

    if is_supernet:
        configs = all_cell_configs(nbr_cells, nbr_choice_blocks)
        scores = []
        losses = []
        for config in configs:
            accs = []
            conf_loss =[]
            for imgs, targets in tqdm.tqdm(dataloader_val, desc="Evaluating on val"):
                batch = imgs.to(device)
                with torch.no_grad():
                    outputs, descs = model(imgs, config)
                    conf_loss.append(loss_fn(outputs, targets))
                    preds = torch.argmax(softmax(outputs), dim=1)
                    accs.append(accuracy(preds, targets))

            acc = sum(accs) / len(accs)
            print(f'Config {config} accuracy {acc}')
            scores.append((config, acc))
            losses.append((config, sum(conf_loss) / len(conf_loss)))

        return scores, losses

    else:
        accs = []
        for imgs, targets in tqdm.tqdm(dataloader_val, desc="Evaluating on val"):
            batch = imgs.to(device)
            with torch.no_grad():
                if passed_cfg:
                    outputs, descs = model(imgs, passed_cfg)
                else:
                    outputs, descs = model(imgs)
                preds = torch.argmax(softmax(outputs), dim=1)
                accs.append(accuracy(preds, targets))

        acc = sum(accs) / len(accs)
        print(f'Accuracy : {acc}')
        return acc

