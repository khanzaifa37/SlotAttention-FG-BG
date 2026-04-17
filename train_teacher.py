"""Teacher-stage training for the Indicator model."""

import argparse
import copy
import math
import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from FB_Indicator import Indicator
from datasets import (COCO2017, COCO2017Teacher, PascalVOC, PascalVOCTeacher,
                      TumorDataset, TumorDatasetTeacher)
from ocl_metrics import ARIMetric, UnsupervisedMaskIoUMetric
from utils_spot import bool_flag, cosine_scheduler, load_pretrained_encoder
import models_vit


def is_running_in_colab():
    return "google.colab" in sys.modules or "COLAB_GPU" in os.environ


def resolve_device(device_arg):
    if isinstance(device_arg, int):
        device_arg = str(device_arg)

    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device(device_arg)
    if device_arg.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA device index was provided but CUDA is not available.")
        return torch.device(f"cuda:{device_arg}")
    raise ValueError(f"Unsupported device value: {device_arg}")


def get_args_parser():
    parser = argparse.ArgumentParser("Indicator teacher training", add_help=False)

    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--val_image_size", type=int, default=224)
    parser.add_argument("--val_mask_size", type=int, default=320)
    parser.add_argument("--clip", type=float, default=0.3)

    parser.add_argument("--dataset", default="coco", help="coco or voc")
    parser.add_argument("--data_path", type=str, required=True, help="dataset path")
    parser.add_argument("--log_path", default="teacher_logs")
    parser.add_argument("--resume_path", default="", help="resume checkpoint if it exists")

    parser.add_argument("--lr_main", type=float, default=4e-5)
    parser.add_argument("--lr_min", type=float, default=4e-7)
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--num_dec_blocks", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--num_slots", type=int, default=2)
    parser.add_argument("--slot_size", type=int, default=256)
    parser.add_argument("--mlp_hidden_size", type=int, default=1024)
    parser.add_argument("--img_channels", type=int, default=3)
    parser.add_argument("--pos_channels", type=int, default=4)
    parser.add_argument("--num_cross_heads", type=int, default=None)
    parser.add_argument("--dec_type", type=str, default="transformer")
    parser.add_argument("--cappa", type=float, default=-1)
    parser.add_argument("--mlp_dec_hidden", type=int, default=2048)
    parser.add_argument("--use_slot_proj", type=bool_flag, default=True)

    parser.add_argument("--which_encoder", type=str, default="dino_vitb16")
    parser.add_argument("--finetune_blocks_after", type=int, default=8)
    parser.add_argument("--encoder_final_norm", type=bool_flag, default=False)
    parser.add_argument("--pretrained_encoder_weights", type=str, default=None)
    parser.add_argument("--use_second_encoder", type=bool_flag, default=False)

    parser.add_argument("--truncate", type=str, default="bi-level")
    parser.add_argument("--init_method", default="mu_embedding")
    parser.add_argument("--train_permutations", type=str, default="standard")
    parser.add_argument("--eval_permutations", type=str, default="standard")
    parser.add_argument("--min-scale", type=float, default=0.08)
    parser.add_argument("--mask_ext", type=str, default=".png",
                        help="Mask file extension for tumor dataset (default: .png)")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Root of the validation split for --dataset tumor. "
                             "Expects val_data_path/images/ and val_data_path/masks/. "
                             "Falls back to data_path if not set.")

    parser.add_argument("--group-loss-weight", default=0.5, type=float)
    parser.add_argument("--ctr-loss-weight", default=0.2, type=float)
    parser.add_argument("--differ_loss_weight", default=0.5, type=float)
    parser.add_argument("--kernel_size", default=1, type=int)
    parser.add_argument("--top_k", default=7, type=int)
    parser.add_argument("--teacher-temp", default=0.07, type=float)
    parser.add_argument("--student-temp", default=0.1, type=float)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--teacher-momentum", default=0.99, type=float)
    parser.add_argument("--center-momentum", default=0.9, type=float)

    parser.add_argument("--device", type=str, default="auto")

    return parser


def load_encoder(args):
    encoder_s = None
    if args.which_encoder == "dino_vitb16":
        args.max_tokens = int((args.image_size / 16) ** 2)
        encoder = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
    elif args.which_encoder == "dino_vits8":
        args.max_tokens = int((args.image_size / 8) ** 2)
        encoder = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
    elif args.which_encoder == "dino_vitb8":
        args.max_tokens = int((args.image_size / 8) ** 2)
        encoder = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
    elif args.which_encoder == "dinov2_vitb14":
        args.max_tokens = int((args.image_size / 14) ** 2)
        encoder = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vitb14")
    elif args.which_encoder == "dinov2_vits14":
        args.max_tokens = int((args.image_size / 14) ** 2)
        encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    elif args.which_encoder == "dinov2_vitb14_reg":
        args.max_tokens = int((args.image_size / 14) ** 2)
        encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    elif args.which_encoder == "dinov2_vits14_reg":
        args.max_tokens = int((args.image_size / 14) ** 2)
        encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
    elif args.which_encoder == "mae_vitb16":
        args.max_tokens = int((args.image_size / 16) ** 2)
        encoder = models_vit.__dict__["vit_base_patch16"](num_classes=0, global_pool=False, drop_path_rate=0)
        assert args.pretrained_encoder_weights is not None
        load_pretrained_encoder(encoder, args.pretrained_encoder_weights, prefix=None)
    else:
        raise ValueError(f"Unsupported encoder: {args.which_encoder}")

    if encoder_s is None:
        encoder_s = copy.deepcopy(encoder)
    return encoder.eval(), encoder_s


def build_datasets(args):
    if args.dataset == "voc":
        train_dataset = PascalVOCTeacher(
            root=args.data_path, split="trainaug", image_size=args.image_size, min_scale=args.min_scale
        )
        val_dataset = PascalVOC(
            root=args.data_path, split="val", image_size=args.val_image_size, mask_size=args.val_mask_size
        )
    elif args.dataset == "coco":
        train_dataset = COCO2017Teacher(
            root=args.data_path, split="train", image_size=args.image_size, min_scale=args.min_scale
        )
        val_dataset = COCO2017(
            root=args.data_path, split="val", image_size=args.val_image_size, mask_size=args.val_mask_size
        )
    elif args.dataset == "tumor":
        # data_path/images/  → teacher training images (unsupervised, no masks needed)
        # val_data_path/images/ + val_data_path/masks/ → validation with ground-truth masks
        # If --val_data_path is not provided it falls back to data_path.
        train_images_dir = os.path.join(args.data_path, "images")
        val_root   = args.val_data_path if args.val_data_path else args.data_path
        val_images = os.path.join(val_root, "images")
        val_masks  = os.path.join(val_root, "masks")
        train_dataset = TumorDatasetTeacher(
            images_dir=train_images_dir, image_size=args.image_size, min_scale=args.min_scale
        )
        val_dataset = TumorDataset(
            images_dir=val_images, masks_dir=val_masks, split="val",
            image_size=args.val_image_size, mask_size=args.val_mask_size,
            mask_ext=args.mask_ext,
        )
    else:
        raise ValueError("--dataset must be one of: coco, voc, tumor")
    return train_dataset, val_dataset


def move_teacher_batch(batch, device):
    crops, coords, flags = batch
    crops = [tensor.to(device, non_blocking=True) for tensor in crops]
    coords = [tensor.to(device, non_blocking=True) for tensor in coords]
    flags = [tensor.to(device, non_blocking=True) for tensor in flags]
    return crops, coords, flags


def train_teacher(args):
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if args.num_workers is None:
        args.num_workers = 0 if is_running_in_colab() else 4

    os.makedirs(args.log_path, exist_ok=True)
    log_dir = os.path.join(args.log_path, datetime.today().isoformat())
    os.makedirs(log_dir, exist_ok=True)

    train_dataset, val_dataset = build_datasets(args)
    args.num_instances = len(train_dataset)

    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size, **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, batch_size=args.eval_batch_size, **loader_kwargs
    )

    if len(train_loader) == 0:
        raise ValueError("Training loader is empty. Increase dataset size or lower batch_size.")

    encoder, encoder_s = load_encoder(args)
    encoder_second = copy.deepcopy(encoder).eval() if args.use_second_encoder else None
    if args.num_cross_heads is None:
        args.num_cross_heads = args.num_heads

    model = Indicator(encoder_s, args, encoder_second).to(device)
    optimizer = Adam(
        [{"params": (param for param in model.parameters() if param.requires_grad), "lr": args.lr_main}],
        weight_decay=args.weight_decay,
    )

    writer = SummaryWriter(log_dir)
    writer.add_text("hparams", "__".join([f"{k}={v}" for k, v in vars(args).items()]))

    start_epoch = 0
    best_miou = 0.0
    best_epoch = 0
    if args.resume_path and os.path.isfile(args.resume_path):
        checkpoint = torch.load(args.resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0)
        best_miou = checkpoint.get("best_miou", 0.0)
        best_epoch = checkpoint.get("best_epoch", 0)

    lr_schedule = cosine_scheduler(
        base_value=args.lr_main,
        final_value=args.lr_min,
        epochs=args.epochs,
        niter_per_ep=len(train_loader),
        warmup_epochs=int(args.lr_warmup_steps / max(len(train_dataset) / args.batch_size, 1)),
        start_warmup_value=0,
    )

    mbo_c_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background=True, ignore_overlaps=True).to(device)
    mbo_i_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background=True, ignore_overlaps=True).to(device)
    miou_metric = UnsupervisedMaskIoUMetric(matching="hungarian", ignore_background=True, ignore_overlaps=True).to(device)
    ari_metric = ARIMetric(foreground=True, ignore_overlaps=True).to(device)

    log_interval = max(1, len(train_loader) // 5)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            crops, coords, flags = move_teacher_batch(batch, device)
            global_step = epoch * len(train_loader) + batch_idx
            optimizer.param_groups[0]["lr"] = lr_schedule[global_step]

            optimizer.zero_grad()
            loss, group_loss, ctr_loss, differ_loss, _ = model((crops, coords, flags))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip, "inf")
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {:3} [{:5}/{:5}] \t lr = {:5} \t total_loss: {:F} \t group_loss: {:F} \t ctr_loss: {:F} \t differ_loss: {:F}".format(
                        epoch + 1,
                        batch_idx,
                        len(train_loader),
                        optimizer.param_groups[0]["lr"],
                        loss.item(),
                        group_loss.item(),
                        ctr_loss.item(),
                        differ_loss.item(),
                    )
                )
                writer.add_scalar("TRAIN/total_loss", loss.item(), global_step)
                writer.add_scalar("TRAIN/group_loss", group_loss.item(), global_step)
                writer.add_scalar("TRAIN/ctr_loss", ctr_loss.item(), global_step)
                writer.add_scalar("TRAIN/differ_loss", differ_loss.item(), global_step)

        model.eval()
        with torch.no_grad():
            for image, true_mask_i, true_mask_c, mask_ignore in tqdm(val_loader):
                image = image.to(device)
                true_mask_i = true_mask_i.to(device)
                true_mask_c = true_mask_c.to(device)
                mask_ignore = mask_ignore.to(device)

                _, slots_attns, _, _, _, _ = model.forward_eval(image)
                pred_mask = F.interpolate(slots_attns, size=args.val_mask_size, mode="bilinear").argmax(1)

                true_mask_i_reshaped = F.one_hot(true_mask_i).to(torch.float32).permute(0, 3, 1, 2).to(device)
                true_mask_c_reshaped = F.one_hot(true_mask_c).to(torch.float32).permute(0, 3, 1, 2).to(device)
                pred_mask_reshaped = F.one_hot(pred_mask).to(torch.float32).permute(0, 3, 1, 2).to(device)

                mbo_i_metric.update(pred_mask_reshaped, true_mask_i_reshaped, mask_ignore)
                mbo_c_metric.update(pred_mask_reshaped, true_mask_c_reshaped, mask_ignore)
                miou_metric.update(pred_mask_reshaped, true_mask_i_reshaped, mask_ignore)
                ari_metric.update(pred_mask_reshaped, true_mask_i_reshaped, mask_ignore)

            ari = 100 * ari_metric.compute()
            mbo_c = 100 * mbo_c_metric.compute()
            mbo_i = 100 * mbo_i_metric.compute()
            miou = 100 * miou_metric.compute()

            writer.add_scalar("VAL/ari", ari, epoch + 1)
            writer.add_scalar("VAL/mbo_c", mbo_c, epoch + 1)
            writer.add_scalar("VAL/mbo_i", mbo_i, epoch + 1)
            writer.add_scalar("VAL/miou", miou, epoch + 1)

            print(
                "====> Epoch: {:3} \t ARI = {:F} \t mBO_c = {:F} \t mBO_i = {:F} \t miou = {:F}".format(
                    epoch + 1, ari, mbo_c, mbo_i, miou
                )
            )

            ari_metric.reset()
            mbo_c_metric.reset()
            mbo_i_metric.reset()
            miou_metric.reset()

            if miou > best_miou:
                best_miou = miou
                best_epoch = epoch + 1
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "best_miou": best_miou,
                        "best_epoch": best_epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(log_dir, "best_model.pt"),
                )

            checkpoint = {
                "epoch": epoch + 1,
                "best_miou": best_miou,
                "best_epoch": best_epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(log_dir, "checkpoint.pt.tar"))
            print("====> Best mIoU = {:F} @ Epoch {}".format(best_miou, best_epoch))

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Indicator teacher training", parents=[get_args_parser()])
    train_teacher(parser.parse_args())
