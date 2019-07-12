import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from SuperPoint.dataset.tum_dataset import TUMDataset, collate as tum_collate
from SuperPoint.model.super_point import SuperPoint

from common.model_utils import detector_loss, descriptor_loss, detector_metrics, filter_probabilities
from common.utils import *


def launch_training():
    print("Starting preparations...")

    # config = load_config('configs/config_check.yaml')
    config = load_config('configs/config.yaml')
    data_config = config['data']
    model_config = config['model']
    experiment_config = config['experiment']

    set_seed(experiment_config['seed'])

    print("Loading data...")

    train_dataset = TUMDataset(TRAINING, data_config)
    val_dataset = TUMDataset(VALIDATION, data_config, train_ratio=0.9)

    train_data_loader = DataLoader(train_dataset, model_config['batch_size'], collate_fn=tum_collate, shuffle=True)
    val_data_loader = DataLoader(val_dataset, model_config['eval_batch_size'], collate_fn=tum_collate, shuffle=True)

    # device = torch.device('cpu')
    device = torch.device(type='cuda', index=int(os.environ["CUDA_VISIBLE_DEVICES"]))

    epoch = 0
    model = SuperPoint(model_config).to(device)
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

        train_total_det_loss = 0
        train_det_loss = 0
        train_warped_det_loss = 0

        train_desc_loss = 0

        train_det_precision = 0
        train_det_recall = 0

        train_warped_det_precision = 0
        train_warped_det_recall = 0

        for item in train_data_loader:
            optimizer.zero_grad()

            y_pred = model(item[IMAGE].to(device))
            y_warped_pred = model(item[WARPED_IMAGE].to(device))

            det_loss = detector_loss(y_pred['logits'].to(device), item[KEYPOINT_MAP].to(device),
                                     item[MASK].to(device), device, model_config)
            warped_det_loss = detector_loss(y_warped_pred['logits'].to(device),
                                            item[WARPED_KEYPOINT_MAP].to(device),
                                            item[WARPED_MASK].to(device), device, model_config)

            total_det_loss = det_loss + warped_det_loss

            desc_loss = model_config['lambda_loss'] * descriptor_loss(y_pred['raw_desc'], y_warped_pred['raw_desc'],
                                                                      item[HOMOGRAPHY], item[WARPED_MASK].to(device),
                                                                      device,
                                                                      model_config)

            loss = total_det_loss + desc_loss

            loss.backward()
            optimizer.step()

            probs = filter_probabilities(y_pred['probs'], model_config)
            warped_probs = filter_probabilities(y_warped_pred['probs'], model_config)

            metrics = detector_metrics(probs, item[KEYPOINT_MAP].to(device))
            warped_metrics = detector_metrics(warped_probs, item[WARPED_KEYPOINT_MAP].to(device))

            train_total_det_loss += total_det_loss.cpu().item()
            train_det_loss += det_loss.cpu().item()
            train_warped_det_loss += warped_det_loss.cpu().item()

            train_desc_loss += desc_loss.cpu().item()

            train_det_precision += metrics['precision'].cpu().item()
            train_det_recall += metrics['recall'].cpu().item()

            train_warped_det_precision += warped_metrics['precision'].cpu().item()
            train_warped_det_recall += warped_metrics['recall'].cpu().item()

        writer.add_scalar('training/total_det_loss', train_total_det_loss, epoch)
        writer.add_scalar('training/det_loss', train_det_loss, epoch)
        writer.add_scalar('training/warped_det_loss', train_warped_det_loss, epoch)

        writer.add_scalar('training/desc_loss', train_desc_loss, epoch)

        writer.add_scalar('training/det_precision', train_det_precision, epoch)
        writer.add_scalar('training/det_recall', train_det_recall, epoch)

        writer.add_scalar('training/warped_det_precision', train_warped_det_precision, epoch)
        writer.add_scalar('training/warped_det_recall', train_warped_det_recall, epoch)

        model.eval()

        with torch.no_grad():
            val_total_det_loss = 0
            val_det_loss = 0
            val_warped_det_loss = 0

            val_desc_loss = 0

            val_det_precision = 0
            val_det_recall = 0

            val_warped_det_precision = 0
            val_warped_det_recall = 0

            for item in val_data_loader:
                y_pred = model(item[IMAGE].to(device))
                y_warped_pred = model(item[WARPED_IMAGE].to(device))

                det_loss = detector_loss(y_pred['logits'].to(device), item[KEYPOINT_MAP].to(device),
                                         item[MASK].to(device), device, model_config)
                warped_det_loss = detector_loss(y_warped_pred['logits'].to(device),
                                                item[WARPED_KEYPOINT_MAP].to(device),
                                                item[WARPED_MASK].to(device), device, model_config)

                total_det_loss = det_loss + warped_det_loss

                desc_loss = model_config['lambda_loss'] * descriptor_loss(y_pred['raw_desc'], y_warped_pred['raw_desc'],
                                                                          item[HOMOGRAPHY],
                                                                          item[WARPED_MASK].to(device),
                                                                          device,
                                                                          model_config)

                probs = filter_probabilities(y_pred['probs'], model_config)
                warped_probs = filter_probabilities(y_warped_pred['probs'], model_config)

                metrics = detector_metrics(probs, item[KEYPOINT_MAP].to(device))
                warped_metrics = detector_metrics(warped_probs, item[WARPED_KEYPOINT_MAP].to(device))

                val_total_det_loss += total_det_loss.cpu().item()
                val_det_loss += det_loss.cpu().item()
                val_warped_det_loss += warped_det_loss.cpu().item()

                val_desc_loss += desc_loss.cpu().item()

                val_det_precision += metrics['precision'].cpu().item()
                val_det_recall += metrics['recall'].cpu().item()

                val_warped_det_precision += warped_metrics['precision'].cpu().item()
                val_warped_det_recall += warped_metrics['recall'].cpu().item()

            writer.add_scalar('validation/total_det_loss', val_total_det_loss, epoch)
            writer.add_scalar('validation/det_loss', val_det_loss, epoch)
            writer.add_scalar('validation/warped_det_loss', val_warped_det_loss, epoch)

            writer.add_scalar('validation/desc_loss', val_desc_loss, epoch)
            writer.add_scalar('validation/det_precision', val_det_precision, epoch)
            writer.add_scalar('validation/det_recall', val_det_recall, epoch)

            writer.add_scalar('validation/warped_det_precision', val_warped_det_precision, epoch)
            writer.add_scalar('validation/warped_det_recall', val_warped_det_recall, epoch)

        if experiment_config['keep_checkpoints'] != 0 and epoch != 0 and epoch % experiment_config[
            'save_interval'] == 0:
            checkpoint_path = get_checkpoint_path(experiment_config, model_config, epoch)
            save_checkpoint(epoch, model, optimizer, checkpoint_path)
            clear_old_checkpoints(experiment_config)

    writer.close()

    print("Training finished")


if __name__ == "__main__":
    launch_training()
