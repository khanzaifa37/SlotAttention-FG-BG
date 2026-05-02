"""
Standalone evaluation script for running the pretrained ContextFusion model
on a custom tumor-segmentation dataset.

Usage (baseline — pretrained weights, no fine-tuning)
-----------------------------------------------------
python eval_tumor.py \
    --images_dir /path/to/tumor/images \
    --masks_dir  /path/to/tumor/masks \
    --student_checkpoint_path ContextFusion-and-Bootstrap/spot_coco_checkpoint.pt \
    --teacher_checkpoint_path /path/to/teacher_checkpoint.pt \
    --num_slots 7

If you do not have a teacher checkpoint yet, add:
    --teacher_free_smoke_test true
(results will be weaker but the script will still run end-to-end)
"""

import os
import sys
import copy
import math
import argparse
import tempfile
import zipfile

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from contextfusion_bootstrp import SPOT
from FB_Indicator import Indicator
from datasets import TumorDataset
from ocl_metrics import UnsupervisedMaskIoUMetric, ARIMetric
from utils_spot import bool_flag, load_pretrained_encoder
import models_vit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_device(device_arg):
    if device_arg == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available.')
        return torch.device(device_arg)
    raise ValueError(f'Unsupported device: {device_arg}')


def compute_best_dice(pred_mask, gt_instance_mask, num_slots):
    """
    Per-image best-slot Dice against the binary GT tumor mask.

    For each image, tries every predicted slot and picks the one whose
    binary mask best overlaps with the GT tumor region.  Returns the
    average Dice across images that actually contain a tumor.

    pred_mask        : [B, H, W] long  — argmax over slots (0..K-1)
    gt_instance_mask : [B, H, W] long  — GT instance IDs (0 = background)
    num_slots        : int
    """
    B = pred_mask.shape[0]
    gt_binary = gt_instance_mask > 0          # [B, H, W] bool
    scores = []
    for b in range(B):
        gt = gt_binary[b]
        if not gt.any():
            continue                           # skip images with no tumor
        best = 0.0
        for s in range(num_slots):
            pred_s = pred_mask[b] == s         # [H, W] bool
            tp     = (pred_s & gt).sum().float()
            denom  = pred_s.sum().float() + gt.sum().float()
            if denom > 0:
                best = max(best, (2 * tp / denom).item())
        scores.append(best)
    return sum(scores) / len(scores) if scores else 0.0


def load_checkpoint(path, map_location='cpu'):
    """Load a regular .pt/.tar file OR a distributed checkpoint directory."""
    if os.path.isdir(path):
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            with zipfile.ZipFile(tmp.name, mode='w',
                                 compression=zipfile.ZIP_STORED) as arc:
                archive_root = os.path.basename(os.path.normpath(path))
                for root, _, files in os.walk(path):
                    for filename in sorted(files):
                        full = os.path.join(root, filename)
                        rel  = os.path.relpath(full, path)
                        arc.write(full, os.path.join(archive_root, rel))
            return torch.load(tmp.name, map_location=map_location, weights_only=False)
    return torch.load(path, map_location=map_location, weights_only=False)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser('ContextFusion Tumor Evaluation')

    # --- Dataset ---
    p.add_argument('--images_dir',  required=True,
                   help='Folder of tumour images  (jpg / png / tif)')
    p.add_argument('--masks_dir',   required=True,
                   help='Folder of segmentation masks (same stem as images)')
    p.add_argument('--mask_ext',       default='.png',
                   help='Extension of mask files (default: .png)')
    p.add_argument('--mask_threshold', type=int, default=1,
                   help='Pixels >= this value are treated as tumor foreground. '
                        'Use 1 for standard binary masks, 128 or 255 for BRISC '
                        'soft-boundary masks to keep only the high-confidence core.')
    p.add_argument('--image_size',      type=int, default=224)
    p.add_argument('--val_image_size',  type=int, default=224)
    p.add_argument('--val_mask_size',   type=int, default=512,
                   help='Resolution to evaluate masks at. Set to match original '
                        'mask resolution (BRISC=512).')
    p.add_argument('--eval_batch_size', type=int, default=16)
    p.add_argument('--num_workers',     type=int, default=4)

    # --- Encoder ---
    p.add_argument('--which_encoder', default='dino_vitb16',
                   choices=['dino_vitb16', 'dino_vits8', 'dino_vitb8',
                            'dinov2_vitb14', 'dinov2_vits14',
                            'dinov2_vitb14_reg', 'dinov2_vits14_reg',
                            'mae_vitb16'])
    p.add_argument('--pretrained_encoder_weights', default=None,
                   help='Path to MAE weights (required only for mae_vitb16)')
    p.add_argument('--finetune_blocks_after', type=int, default=100,
                   help='Set to 100 to keep entire encoder frozen during eval')
    p.add_argument('--encoder_final_norm', type=bool_flag, default=False)

    # --- Slot Attention ---
    p.add_argument('--num_slots',      type=int, default=7)
    p.add_argument('--slot_size',      type=int, default=256)
    p.add_argument('--num_iterations', type=int, default=3)
    p.add_argument('--mlp_hidden_size',type=int, default=1024)
    p.add_argument('--pos_channels',   type=int, default=4)
    p.add_argument('--truncate',       default='bi-level')
    p.add_argument('--init_method',    default='shared_gaussian')
    p.add_argument('--train_permutations', default='standard')
    p.add_argument('--eval_permutations',  default='standard')

    # --- Decoder ---
    p.add_argument('--dec_type',        default='transformer')
    p.add_argument('--num_dec_blocks',  type=int, default=4)
    p.add_argument('--d_model',         type=int, default=768)
    p.add_argument('--num_heads',       type=int, default=6)
    p.add_argument('--num_cross_heads', type=int, default=None)
    p.add_argument('--dropout',         type=float, default=0.0)
    p.add_argument('--cappa',           type=float, default=-1)
    p.add_argument('--mlp_dec_hidden',  type=int, default=2048)
    p.add_argument('--use_slot_proj',   type=bool_flag, default=True)
    p.add_argument('--use_second_encoder', type=bool_flag, default=False)
    p.add_argument('--top_k',           type=int, default=7)
    p.add_argument('--img_channels',    type=int, default=3)

    # --- Checkpoints ---
    p.add_argument('--student_checkpoint_path', required=True,
                   help='Path to pretrained student checkpoint '
                        '(file or directory, e.g. spot_coco_checkpoint.pt)')
    p.add_argument('--teacher_checkpoint_path', default=None,
                   help='Path to teacher (Indicator) checkpoint')
    p.add_argument('--teacher_free_smoke_test', type=bool_flag, default=False,
                   help='Skip teacher; use student slots as teacher proxy '
                        '(weaker results but no teacher checkpoint needed)')

    # --- Teacher-specific args (mirror train_teacher.py defaults) ---
    p.add_argument('--teacher_truncate',           default='bi-level')
    p.add_argument('--teacher_init_method',        default='mu_embedding')
    p.add_argument('--teacher_train_permutations', default='standard')
    p.add_argument('--teacher_eval_permutations',  default='standard')
    p.add_argument('--teacher-temp',      type=float, default=0.07)
    p.add_argument('--student-temp',      type=float, default=0.1)
    p.add_argument('--center-momentum',   type=float, default=0.9)
    p.add_argument('--teacher-momentum',  type=float, default=0.99)
    p.add_argument('--group-loss-weight', type=float, default=0.5)
    p.add_argument('--ctr-loss-weight',   type=float, default=0.2)
    p.add_argument('--differ_loss_weight',type=float, default=0.5)
    p.add_argument('--kernel_size',       type=int,   default=1)

    # --- Misc ---
    p.add_argument('--device', default='auto')
    p.add_argument('--seed',   type=int, default=0)
    p.add_argument('--num_slots_finetuning', type=int, default=7)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    if args.num_cross_heads is None:
        args.num_cross_heads = args.num_heads

    # ------------------------------------------------------------------ #
    # 1. Dataset                                                           #
    # ------------------------------------------------------------------ #
    val_dataset = TumorDataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        split='val',
        image_size=args.val_image_size,
        mask_size=args.val_mask_size,
        mask_ext=args.mask_ext,
        mask_threshold=args.mask_threshold,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    print(f'Validation set: {len(val_dataset)} images')

    # ------------------------------------------------------------------ #
    # 2. Encoder                                                           #
    # ------------------------------------------------------------------ #
    if args.which_encoder == 'dino_vitb16':
        args.max_tokens = int((args.val_image_size / 16) ** 2)
        encoder   = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        encoder_s = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    elif args.which_encoder == 'dino_vits8':
        args.max_tokens = int((args.val_image_size / 8) ** 2)
        encoder   = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        encoder_s = copy.deepcopy(encoder)
    elif args.which_encoder == 'dino_vitb8':
        args.max_tokens = int((args.val_image_size / 8) ** 2)
        encoder   = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        encoder_s = copy.deepcopy(encoder)
    elif args.which_encoder == 'dinov2_vitb14':
        args.max_tokens = int((args.val_image_size / 14) ** 2)
        encoder   = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitb14')
        encoder_s = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitb14')
    elif args.which_encoder in ('dinov2_vits14', 'dinov2_vitb14_reg', 'dinov2_vits14_reg'):
        args.max_tokens = int((args.val_image_size / 14) ** 2)
        encoder   = torch.hub.load('facebookresearch/dinov2', args.which_encoder)
        encoder_s = copy.deepcopy(encoder)
    elif args.which_encoder == 'mae_vitb16':
        args.max_tokens = int((args.val_image_size / 16) ** 2)
        encoder   = models_vit.__dict__['vit_base_patch16'](
                        num_classes=0, global_pool=False, drop_path_rate=0)
        encoder_s = models_vit.__dict__['vit_base_patch16'](
                        num_classes=0, global_pool=False, drop_path_rate=0)
        assert args.pretrained_encoder_weights, \
            '--pretrained_encoder_weights is required for mae_vitb16'
        load_pretrained_encoder(encoder,   args.pretrained_encoder_weights, prefix=None)
        load_pretrained_encoder(encoder_s, args.pretrained_encoder_weights, prefix=None)
    else:
        raise ValueError(f'Unknown encoder: {args.which_encoder}')

    encoder   = encoder.eval()
    encoder_s = encoder_s.train()

    # ------------------------------------------------------------------ #
    # 3. Student model                                                     #
    # ------------------------------------------------------------------ #
    student_model = SPOT(encoder, encoder_s, args, second_encoder=None)

    ckpt = load_checkpoint(args.student_checkpoint_path, map_location='cpu')

    # Checkpoint may be a raw state-dict or wrapped under 'model'
    state_dict = ckpt.get('model', ckpt)
    # Compatibility: rename old tf_dec. → dec.
    state_dict = {k.replace('tf_dec.', 'dec.'): v for k, v in state_dict.items()}
    msg = student_model.load_state_dict(state_dict, strict=False)
    print('Student checkpoint loaded:', msg)

    student_model = student_model.to(device).eval()

    # ------------------------------------------------------------------ #
    # 4. Teacher model                                                     #
    # ------------------------------------------------------------------ #
    teacher_model = None
    if not args.teacher_free_smoke_test:
        if not args.teacher_checkpoint_path:
            raise ValueError(
                'Provide --teacher_checkpoint_path, '
                'or set --teacher_free_smoke_test true to skip the teacher.'
            )
        args_teacher = copy.deepcopy(args)
        args_teacher.truncate           = args.teacher_truncate
        args_teacher.init_method        = args.teacher_init_method
        args_teacher.train_permutations = args.teacher_train_permutations
        args_teacher.eval_permutations  = args.teacher_eval_permutations
        args_teacher.finetune_blocks_after = 100   # fully frozen
        args_teacher.num_slots          = 2         # FG + BG

        # Indicator.__init__ reads these for momentum-schedule bookkeeping;
        # they are only used during training, not eval, so dummy values are fine.
        args_teacher.num_instances = len(val_dataset)
        args_teacher.batch_size    = args.eval_batch_size
        args_teacher.epochs        = 1
        args_teacher.start_epoch   = 1

        # encoder was moved to GPU by student_model.to(device) above;
        # Indicator.__init__ probes it with a CPU tensor, so copy to CPU first.
        teacher_model = Indicator(copy.deepcopy(encoder).cpu(), args_teacher)

        teacher_ckpt = load_checkpoint(args.teacher_checkpoint_path, map_location='cpu')
        teacher_state = teacher_ckpt.get('model', teacher_ckpt)
        teacher_state = {k.replace('tf_dec.', 'dec.'): v
                         for k, v in teacher_state.items()}
        teacher_model.load_state_dict(teacher_state, strict=True)
        teacher_model = teacher_model.to(device).eval()

        for p in teacher_model.parameters():
            p.requires_grad = False

        print('Teacher checkpoint loaded.')
    else:
        print('Running in teacher-free smoke-test mode.')

    # ------------------------------------------------------------------ #
    # 5. Metrics                                                           #
    # ------------------------------------------------------------------ #
    MBO_c_metric = UnsupervisedMaskIoUMetric(
        matching='best_overlap', ignore_background=True, ignore_overlaps=True).to(device)
    MBO_i_metric = UnsupervisedMaskIoUMetric(
        matching='best_overlap', ignore_background=True, ignore_overlaps=True).to(device)
    miou_metric  = UnsupervisedMaskIoUMetric(
        matching='hungarian',    ignore_background=True, ignore_overlaps=True).to(device)
    ari_metric   = ARIMetric(foreground=True, ignore_overlaps=True).to(device)

    # ------------------------------------------------------------------ #
    # 6. Helper: get teacher targets                                       #
    # ------------------------------------------------------------------ #
    def get_teacher_targets(image):
        if teacher_model is not None:
            return teacher_model.forward_eval(image)
        # Smoke-test fallback: use student encoder + student slots
        emb = student_model.forward_encoder(image, student_model.encoder)
        slots, _, _, _, _ = student_model.slot_attn(emb)
        return emb.detach(), None, slots.detach(), None, None, None

    # ------------------------------------------------------------------ #
    # 7. Evaluation loop                                                   #
    # ------------------------------------------------------------------ #
    val_mse = 0.0
    n_batches = 0
    dice_sum = 0.0
    dice_n_batches = 0

    with torch.no_grad():
        for image, true_mask_i, true_mask_c, mask_ignore in tqdm(val_loader,
                                                                   desc='Evaluating'):
            image       = image.to(device)
            true_mask_i = true_mask_i.to(device)
            true_mask_c = true_mask_c.to(device)
            mask_ignore = mask_ignore.to(device)

            (emb_teacher, _, slots_teacher,
             _, _, _) = get_teacher_targets(image)

            mse, dec_slots_attns, _, _, gamma, beta = \
                student_model.forward_ours_eval(image, emb_teacher, slots_teacher)

            # Upsample decoder attention maps to mask resolution
            dec_attns = F.interpolate(dec_slots_attns,
                                      size=args.val_mask_size,
                                      mode='bilinear')          # [B, K, H, W]
            pred_mask = dec_attns.argmax(dim=1)                 # [B, H, W]

            val_mse += mse.item()
            n_batches += 1

            # Dice: best-slot match against binary GT tumor mask
            batch_dice = compute_best_dice(pred_mask, true_mask_i, dec_attns.shape[1])
            dice_sum += batch_dice
            dice_n_batches += 1

            # One-hot encode for metrics  [B, H, W] → [B, K, 1, H, W]
            pred_oh   = F.one_hot(pred_mask).float().permute(0, 3, 1, 2).to(device)
            true_i_oh = F.one_hot(true_mask_i).float().permute(0, 3, 1, 2).to(device)
            true_c_oh = F.one_hot(true_mask_c).float().permute(0, 3, 1, 2).to(device)

            MBO_i_metric.update(pred_oh, true_i_oh, mask_ignore)
            MBO_c_metric.update(pred_oh, true_c_oh, mask_ignore)
            miou_metric.update( pred_oh, true_i_oh, mask_ignore)
            ari_metric.update(  pred_oh, true_i_oh, mask_ignore)

    # ------------------------------------------------------------------ #
    # 8. Print results                                                     #
    # ------------------------------------------------------------------ #
    ari   = 100 * ari_metric.compute()
    mbo_c = 100 * MBO_c_metric.compute()
    mbo_i = 100 * MBO_i_metric.compute()
    miou  = 100 * miou_metric.compute()
    mse   = val_mse / n_batches
    dice  = 100 * (dice_sum / dice_n_batches if dice_n_batches > 0 else 0.0)

    print('\n' + '=' * 60)
    print('  Evaluation on Tumor Dataset')
    print('=' * 60)
    print(f'  MSE  (reconstruction) : {mse:.6f}')
    print(f'  Dice (best-slot)      : {dice:.2f}')
    print(f'  ARI  (foreground)     : {ari:.2f}')
    print(f'  mBO_i (instance)      : {mbo_i:.2f}')
    print(f'  mBO_c (class)         : {mbo_c:.2f}')
    print(f'  mIoU  (hungarian)     : {miou:.2f}')
    print('=' * 60)

    teacher_mode = 'teacher-free smoke test' if args.teacher_free_smoke_test \
                   else args.teacher_checkpoint_path
    print(f'\nEncoder            : {args.which_encoder}')
    print(f'Num slots          : {args.num_slots}')
    print(f'Student checkpoint : {args.student_checkpoint_path}')
    print(f'Teacher            : {teacher_mode}')


if __name__ == '__main__':
    main()
