import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from common.model_utils import detector_loss, detector_metrics
from common.utils import load_config, set_seed, get_checkpoint_path, save_checkpoint, clear_old_checkpoints, \
    load_checkpoint, get_logs_path

from MagicPoint.model.magic_point import MagicPoint
from MagicPoint.dataset.art_dataset import ArtificialDataset, available_modes, collate, IMAGE, KEYPOINT_MAP, MASK


def launch_training():
    print("Starting preparations...")

    config = load_config('configs/art_config.yaml')

    data_config = config['data']
    model_config = config['model']
    experiment_config = config['experiment']

    set_seed(experiment_config['seed'])

    print("Loading data...")

    train_dataset = ArtificialDataset(available_modes[0], data_config)
    val_dataset = ArtificialDataset(available_modes[1], data_config)

    train_data_loader = DataLoader(train_dataset, model_config['batch_size'], num_workers=4,
                                   collate_fn=collate, shuffle=True)
    val_data_loader = DataLoader(val_dataset, model_config['val_batch_size'], num_workers=2,
                                 collate_fn=collate, shuffle=True)

    device = torch.device(type='cuda', index=int(os.environ["CUDA_VISIBLE_DEVICES"]))

    epoch = 0
    model = MagicPoint(model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=model_config['learning_rate'])

    if experiment_config['load_checkpoints']:
        checkpoint_path = get_checkpoint_path(experiment_config, model_config,
                                              experiment_config['load_checkpoint_iter'])
        if checkpoint_path.exists():
            epoch, model_sd, optimizer_sd = load_checkpoint(checkpoint_path)
            model.load_state_dict(model_sd)
            optimizer.load_state_dict(optimizer_sd)

    writer = SummaryWriter(log_dir=get_logs_path(experiment_config))

    print("Beginning training...")

    for epoch in range(epoch, experiment_config['num_epochs']):

        model.train()

        train_loss = 0
        train_precision = 0
        train_recall = 0

        for item in train_data_loader:
            optimizer.zero_grad()

            y_pred = model(item[IMAGE].to(device))
            loss = detector_loss(y_pred['logits'].to(device), item[KEYPOINT_MAP].to(device), item[MASK].to(device),
                                 device, model_config)

            loss.backward()
            optimizer.step()

            metrics = detector_metrics(y_pred['probs'], item[KEYPOINT_MAP].to(device))

            train_loss += loss.cpu().item()
            train_precision += metrics['precision'].cpu().item()
            train_recall += metrics['recall'].cpu().item()

        train_loss /= train_data_loader.__len__()
        train_precision /= train_data_loader.__len__()
        train_recall /= train_data_loader.__len__()

        writer.add_scalar('training/loss', train_loss, epoch)
        writer.add_scalar('training/precision', train_precision, epoch)
        writer.add_scalar('training/recall', train_recall, epoch)

        model.eval()

        with torch.no_grad():
            val_loss = 0
            val_precision = 0
            val_recall = 0

            for item in val_data_loader:
                y_pred = model(item[IMAGE].to(device))
                loss = detector_loss(y_pred['logits'].to(device), item[KEYPOINT_MAP].to(device), item[MASK].to(device),
                                     device, model_config)

                metrics = detector_metrics(y_pred['probs'], item[KEYPOINT_MAP].to(device))

                val_loss += loss.cpu().item()
                val_precision += metrics['precision'].cpu().item()
                val_recall += metrics['recall'].cpu().item()

            val_loss /= val_data_loader.__len__()
            val_precision /= val_data_loader.__len__()
            val_recall /= val_data_loader.__len__()

            writer.add_scalar('validation/loss', val_loss, epoch)
            writer.add_scalar('validation/precision', val_precision, epoch)
            writer.add_scalar('validation/recall', val_recall, epoch)

        if experiment_config['keep_checkpoints'] != 0 and epoch != 0 and epoch % experiment_config['save_interval'] == 0:
            checkpoint_path = get_checkpoint_path(experiment_config, model_config, epoch)
            save_checkpoint(epoch, model, optimizer, checkpoint_path)
            clear_old_checkpoints(experiment_config)

    writer.close()

    print("Training finished")


if __name__ == "__main__":
    launch_training()
