#! /usr/bin/env python3
import argparse
import random
import numpy as np
import os
import tqdm
import wandb

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from src.models.model import load_model, sample_from_supernet
from src.datasets.cifar10 import get_cifar_dataloader
from src.utils.utils import random_cell_config
from src.utils.metrics import accuracy
from test import evaluate


def train(train_args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if train_args.sampled_architecture_config:
        sa_config = [int(c) for c in train_args.sampled_architecture_config]
        print(sa_config)
        model, _, __ = sample_from_supernet(train_args.model, sa_config)
    else:
        model = load_model(train_args.model, train_args.pretrained_weights, device)

    nbr_cells = model.hyperparams['nbr_cells']
    nbr_choice_blocks = model.hyperparams['nbr_choice_blocks']

    print(nbr_cells)
    print(nbr_choice_blocks)

    wandb.login()
    run = wandb.init(
        project="default-sampled[1, 1]-training",
        config={
            "Cells_n_blocks": [nbr_cells, nbr_choice_blocks],
            "learning_rate": model.hyperparams['lr'],
            "epochs": train_args.epochs,
            "transforms": 'RandomHorizontalFlip'
        })

    transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])])

    transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])])

    dataloader_train = get_cifar_dataloader(train_args.batch_size,
                                            transforms_train, is_train=True,
                                            num_workers=train_args.num_workers)

#   TODO: split train into train and val to prevent architectural bias

    dataloader_val = get_cifar_dataloader(train_args.batch_size,
                                          transforms_val, is_train=False,
                                          num_workers=train_args.num_workers)

    parameters = [param for param in model.parameters() if param.requires_grad]
#   Use SGD with momentum same as in the paper (Bender 2018)
    optimizer = torch.optim.SGD(
                                parameters,
                                lr=model.hyperparams['lr'],
                                weight_decay=model.hyperparams['decay'],
                                momentum=model.hyperparams['momentum']
    )
    loss_fn = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    print(f'\n START TRAINING FOR {train_args.epochs} epochs')

    for epoch in range(train_args.start_epoch, train_args.epochs + 1):

        model.train(True)
        # TODO add normal lr scheduler + warm-up option for models with pretrained parts
        if epoch % 5 == 0:
            lr = model.hyperparams['lr']
            lr = lr * 0.1
            for g in optimizer.param_groups:
                g['lr'] = lr

        # mean of random architectures during one epoch
        avg_train_accs = []
        for i, (imgs, targets) in enumerate(tqdm.tqdm(dataloader_train, desc=f'Epoch {epoch}')):
            batch = imgs.to(device)

            if train_args.is_supernet:
                cell_config = random_cell_config(nbr_cells,
                                                 nbr_choice_blocks)
            elif train_args.sampled_architecture_config:
                cell_config = [int(c) for c in train_args.sampled_architecture_config]
            elif isinstance(nbr_choice_blocks, list):
                cell_config = [nbr_choice_blocks[i] for i in range(nbr_cells)]
            else:
                cell_config = [nbr_choice_blocks for _ in range(nbr_cells)]

            outputs, arch_desc_list = model(batch, cell_config)

            preds = torch.argmax(softmax(outputs), dim=1)
            avg_train_accs.append(accuracy(preds, targets))

            loss = loss_fn(outputs, targets)
            loss.backward()
            # TODO add lr scheduler
            optimizer.step()
            optimizer.zero_grad()

        if epoch % train_args.checkpoint_interval == 0:
            # TODO add optimizer state
            torch.save(model.state_dict(),
                       f'{train_args.checkpoint_path}supernet_{epoch}.pt')
            print(f'Checkpoint saved')

        if epoch % train_args.evaluation_interval == 0:
            print("Evaluation \n")

            if train_args.is_supernet:
                scores, losses = evaluate(model, dataloader_val, nbr_cells, nbr_choice_blocks,
                                          device, is_supernet=train_args.is_supernet)

                print(f'Epoch {epoch} metrics: {scores}')
                wandb.log({f'val_{k}': v for k, v in scores})
                val_accs_values = [v for k, v in scores]
                wandb.log({'average_val_accuracy': sum(val_accs_values) / len(val_accs_values)})
                val_losses_values = [v for k, v in losses]
                wandb.log({'average_val_loss': sum(val_losses_values) / len(val_losses_values)})
            else:
                acc = evaluate(model, dataloader_val, nbr_cells, nbr_choice_blocks,
                               device, passed_cfg=cell_config,
                               is_supernet=train_args.is_supernet)
                wandb.log({'val_accuracy': acc})

        print(f'Epoch {epoch} loss: {loss}')
        wandb.log({'train_loss': loss, 'train_acc': sum(avg_train_accs) / len(avg_train_accs)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for Supernet train args')
    parser.add_argument('--model', type=str, default='src/configs/supernet_config.yaml',
                        help='Path to model config yaml file')
    parser.add_argument('--is_supernet', action='store_true',
                        help='apply random architecture config during training')
    parser.add_argument('-sa', '--sampled_architecture_config', action='append', default=None,
                        help='config list for a stand-alone model')
    parser.add_argument('--pretrained_weights', type=str,
                        help='path to pretrained weights')
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/',
                        help='path to a dir in which to save checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=1,
                        help='nbr of epochs between checkpoint savings')
    parser.add_argument('--evaluation_interval', type=int, default=1,
                        help='nbr of epochs between evaluations')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=-1, help='change to fix distributions')

    args = parser.parse_args()

    if args.seed != -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    os.makedirs(args.checkpoint_path, exist_ok=True)

    train(args)
