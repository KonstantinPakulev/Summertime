import os
import sys

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import time
import random
import argparse
import numpy as np

import torch
import torch.nn
import torch.cuda
import torch.backends.cudnn as cudnn
from torch import autograd
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from legacy.ST_Net.config import cfg
from legacy.ST_Net.data.hpatch_dataset import (
    HPatchDataset,
    Grayscale,
    Normalize,
    Rescale,
    LargerRescale,
    RandomCrop,
    ToTensor,
)
from legacy.ST_Net.model.st_det_vgg import STDetVGGModule
from legacy.ST_Net.model.st_des_vgg import STDesVGGModule
from legacy.ST_Net.model.st_net_vgg import STNetVGGModule
from legacy.ST_Net.utils.common_utils import gct, pretty_dict
from legacy.ST_Net.utils.train_utils import (
    parse_batch,
    parse_unsqueeze,
    mgpu_merge,
    writer_log,
    ExponentialLR,
    SgdLR
)


def Lr_Schechuler(lr_schedule, optimizer, epoch, cfg):
    if lr_schedule == "exp":
        ExponentialLR(optimizer, epoch, cfg)
    elif lr_schedule == "sgd":
        SgdLR(optimizer, cfg)


def select_optimizer(optim, param, lr, wd):
    if optim == "sgd":
        optimizer = torch.optim.SGD(param, lr=lr, momentum=0.9, dampening=0.9, weight_decay=wd)
    elif optim == "adam":
        optimizer = torch.optim.Adam(param, lr=lr, weight_decay=wd)
    else:
        raise Exception(f"Not supported optimizer: {optim}")
    return optimizer


def create_optimizer(optim, model, attr, lr, weight_decay, m_gpu=False):
    param = getattr(model.module, attr).parameters() if m_gpu else getattr(model, attr).parameters()

    return select_optimizer(optim, param, lr, weight_decay)


def parse_parms():
    parser = argparse.ArgumentParser(description="ST-Net")
    parser.add_argument(
        "--resume", default="", type=str, help="latest checkpoint (default: none)"
    )
    parser.add_argument(
        "--save", default="", type=str, help="source code save path (default: none)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_parms()
    cfg.TRAIN.SAVE = args.save

    # Print info
    print(f"{gct()} : Called with args:{args}")
    print(f"{gct()} : Using config:")
    pretty_dict(cfg)

    # Set seed
    print(f"{gct()} : Prepare for repetition")
    device = torch.device("cuda" if cfg.PROJ.USE_GPU else "cpu")
    m_gpu = True if cfg.PROJ.USE_GPU and torch.cuda.device_count() > 1 else False
    seed = cfg.PROJ.SEED

    if cfg.PROJ.USE_GPU:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
        if m_gpu:
            print(f"{gct()} : Train with {torch.cuda.device_count()} GPUs")

    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"{gct()} : Build the model")
    det = STDetVGGModule(cfg.MODEL.GRID_SIZE,
                         cfg.TRAIN.NMS_THRESH,
                         cfg.TRAIN.NMS_KSIZE,
                         cfg.TRAIN.TOPK,
                         cfg.MODEL.GAUSSIAN_KSIZE,
                         cfg.MODEL.GAUSSIAN_SIGMA)
    des = STDesVGGModule(8, 128)
    model = STNetVGGModule(det, des)

    if m_gpu:
        model = torch.nn.DataParallel(model)

    model = model.to(device=device)

    # Load train data
    PPT = [cfg.PROJ.TRAIN_PPT, (cfg.PROJ.TRAIN_PPT + cfg.PROJ.EVAL_PPT)]

    print(f"{gct()} : Loading traning data")
    train_data = DataLoader(
        HPatchDataset(
            data_type="train",
            PPT=PPT,
            use_all=cfg.PROJ.TRAIN_ALL,
            csv_file=cfg[cfg.PROJ.TRAIN]["csv"],
            root_dir=cfg[cfg.PROJ.TRAIN]["root"],
            transform=transforms.Compose(
                [
                    Grayscale(),
                    Normalize(
                        mean=cfg[cfg.PROJ.TRAIN]["MEAN"], std=cfg[cfg.PROJ.TRAIN]["STD"]
                    ),
                    LargerRescale((960, 1280)),
                    RandomCrop((720, 960)),
                    Rescale((240, 320)),
                    ToTensor(),
                ]
            ),
        ),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    # Load eval data
    print(f"{gct()} : Loading evaluation data")
    val_data = DataLoader(
        HPatchDataset(
            data_type="eval",
            PPT=PPT,
            use_all=cfg.PROJ.EVAL_ALL,
            csv_file=cfg[cfg.PROJ.EVAL]["csv"],
            root_dir=cfg[cfg.PROJ.EVAL]["root"],
            transform=transforms.Compose(
                [
                    Grayscale(),
                    Normalize(
                        mean=cfg[cfg.PROJ.EVAL]["MEAN"], std=cfg[cfg.PROJ.EVAL]["STD"]
                    ),
                    Rescale((960, 1280)),
                    Rescale((240, 320)),
                    ToTensor(),
                ]
            ),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # Load test data
    print(f"{gct()} : Loading testing data")
    test_data = DataLoader(
        HPatchDataset(
            data_type="test",
            PPT=PPT,
            use_all=cfg.PROJ.TEST_ALL,
            csv_file=cfg[cfg.PROJ.TEST]["csv"],
            root_dir=cfg[cfg.PROJ.TEST]["root"],
            transform=transforms.Compose(
                [
                    Grayscale(),
                    Normalize(
                        mean=cfg[cfg.PROJ.TEST]["MEAN"], std=cfg[cfg.PROJ.TEST]["STD"]
                    ),
                    Rescale((960, 1280)),
                    Rescale((240, 320)),
                    ToTensor(),
                ]
            ),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    det_optim = create_optimizer(
        cfg.TRAIN.DET_OPTIMIZER,
        model,
        attr='det',
        lr=cfg.TRAIN.DET_LR,
        weight_decay=cfg.TRAIN.DET_WD,
        m_gpu=m_gpu)

    des_optim = create_optimizer(
        cfg.TRAIN.DES_OPTIMIZER,
        model,
        attr='des',
        lr=cfg.TRAIN.DES_LR,
        weight_decay=cfg.TRAIN.DES_WD,
        m_gpu=m_gpu
    )

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"{gct()} : Loading checkpoint {args.resume}")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            det_optim.load_state_dict(checkpoint["det_optim"])
        else:
            print(f"{gct()} : Cannot found checkpoint {args.resume}")
    else:
        args.start_epoch = 0

    train_writer = SummaryWriter(f"{args.save}/log/train")
    test_writer = SummaryWriter(f"{args.save}/log/test")


    def train():
        start_time = time.time()
        for i_batch, sample_batched in enumerate(train_data, 1):
            model.train()
            batch = parse_batch(sample_batched, device)
            with autograd.detect_anomaly():
                model.zero_grad()

                det_optim.zero_grad()
                des_optim.zero_grad()

                endpoint = model(batch)

                _, detloss, desloss = (model.module.criterion(endpoint) if m_gpu else model.criterion(endpoint))

                detloss.backward(retain_graph=True)
                desloss.backward()

                det_optim.step()
                des_optim.step()

            Lr_Schechuler(cfg.TRAIN.DET_LR_SCHEDULE, det_optim, epoch, cfg)
            Lr_Schechuler(cfg.TRAIN.DES_LR_SCHEDULE, des_optim, epoch, cfg)

            # log
            if i_batch % cfg.TRAIN.LOG_INTERVAL == 0 and i_batch > 0:
                elapsed = time.time() - start_time
                model.eval()
                with torch.no_grad():
                    eptr = model(parse_unsqueeze(train_data.dataset[0], device))

                    PLT, cur_detloss, cur_desloss = (
                        model.module.criterion(eptr) if m_gpu else model.criterion(eptr)
                    )

                    PLTS = PLT["scalar"]
                    PLTS["det_lr"] = det_optim.param_groups[0]["lr"]

                    if m_gpu:
                        mgpu_merge(PLTS)

                    iteration = (epoch - 1) * len(train_data) + (i_batch - 1)
                    writer_log(train_writer, PLT["scalar"], iteration)

                    pstring = (
                        "epoch {:2d} | {:4d}/{:4d} batches | ms {:4.02f} | "
                        "sco {:07.05f}".format(
                            epoch,
                            i_batch,
                            len(train_data) // cfg.TRAIN.BATCH_SIZE,
                            elapsed / cfg.TRAIN.LOG_INTERVAL,
                            PLTS["det_loss"]
                        )
                    )

                    ept = model(parse_unsqueeze(val_data.dataset[0], device))
                    PLT, _, _ = (
                        model.module.criterion(ept) if m_gpu else model.criterion(ept)
                    )
                    writer_log(test_writer, PLT["scalar"], iteration)

                    print(f"{gct()} | {pstring}")
                    start_time = time.time()


    print(f"{gct()} : Start training")
    best_f = None
    start_epoch = args.start_epoch + 1
    end = cfg.TRAIN.EPOCH_NUM
    for epoch in range(start_epoch, end):
        epoch_start_time = time.time()
        train()

        # Save the model each 2 epochs
        if epoch % 2:
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "det_optim": det_optim.state_dict(),
            }
            filename = f"{args.save}/model/e{epoch:03d}.pth.tar"
            torch.save(state, filename)
            best_f = filename

        print("-" * 96)
        print(
            "| end of epoch {:3d} | time: {:5.02f}s  ".format(
                epoch, (time.time() - epoch_start_time)
            )
        )
        print("-" * 96)

    # Load the best saved model.
    with open(best_f, "rb") as f:
        model.load_state_dict(torch.load(f)["state_dict"])

    train_writer.close()
    test_writer.close()
