''' Based on SPOT libraries:
https://github.com/gkakogeorgiou/spot.git
'''

import math
import os
import os.path
import argparse
import sys
import tempfile
import zipfile
from tqdm import tqdm
from datetime import datetime
import copy
import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from torch.nn import CrossEntropyLoss
from contextfusion_bootstrp import SPOT
from FB_Indicator import Indicator
from datasets import PascalVOC, COCO2017, MOVi, TumorDataset
from ocl_metrics import UnsupervisedMaskIoUMetric, ARIMetric
from utils_spot import inv_normalize, cosine_scheduler, visualize, att_matching, bool_flag, load_pretrained_encoder
import models_vit
from torchvision.utils import save_image

IGNORE_INDEX = -100


def is_running_in_colab():
    return 'google.colab' in sys.modules or 'COLAB_GPU' in os.environ


def resolve_device(device_arg):
    if isinstance(device_arg, int):
        device_arg = str(device_arg)

    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        return torch.device('cpu')

    if device_arg == 'cpu':
        return torch.device('cpu')

    if device_arg.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available.')
        return torch.device(device_arg)

    if device_arg.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError('A CUDA device index was provided but CUDA is not available.')
        return torch.device(f'cuda:{device_arg}')

    raise ValueError(f'Unsupported device value: {device_arg}')


def load_checkpoint(path, map_location='cpu'):
    if os.path.isdir(path):
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp_file:
            with zipfile.ZipFile(tmp_file.name, mode='w', compression=zipfile.ZIP_STORED) as archive:
                archive_root = os.path.basename(os.path.normpath(path))
                for root, _, files in os.walk(path):
                    for filename in sorted(files):
                        full_path = os.path.join(root, filename)
                        relative_name = os.path.relpath(full_path, path)
                        archive_name = os.path.join(archive_root, relative_name)
                        archive.write(full_path, archive_name)
            return torch.load(tmp_file.name, map_location=map_location, weights_only=False)

    return torch.load(path, map_location=map_location, weights_only=False)

def get_args_parser():
    parser = argparse.ArgumentParser('SPOT (2)', add_help=False)
    
    parser.add_argument('--num_workers', type=int, default=None, help='Defaults to 0 in Colab and 4 elsewhere')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--clip', type=float, default=0.3)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--val_image_size', type=int, default=224)
    parser.add_argument('--val_mask_size', type=int, default=320)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--eval_viz_percent', type=float, default=0.2)
    
    parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar', help='checkpoint to continue the training, loaded only if exists')
    parser.add_argument('--log_path', default='logs')
    parser.add_argument('--dataset', default='coco', help='coco, voc, movi, or tumor')
    parser.add_argument('--data_path',  type=str, help='dataset path')
    parser.add_argument('--val_data_path', type=str, default=None,
                        help='Validation data root for --dataset tumor. '
                             'Expects val_data_path/images/ and val_data_path/masks/. '
                             'Falls back to data_path if not set.')
    parser.add_argument('--mask_ext', type=str, default='.png',
                        help='Mask file extension for tumor dataset (default: .png)')
    parser.add_argument('--mask_threshold', type=int, default=1,
                        help='Pixels >= this value treated as tumor foreground. '
                             'Use 128 or 255 for BRISC soft-boundary masks.')
    parser.add_argument('--predefined_movi_json_paths', default = None,  type=str, help='For MOVi datasets, use the same subsampled images. Typically for the 2nd stage of Spot training to retain the same images')
    
    parser.add_argument('--lr_main', type=float, default=4e-4)
    parser.add_argument('--lr_min', type=float, default=4e-7)
    parser.add_argument('--lr_warmup_steps', type=int, default=10000)
    
    parser.add_argument('--num_dec_blocks', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--num_slots', type=int, default=7)
    parser.add_argument('--num_slots_finetuning', type=int, default=7)
    
    parser.add_argument('--slot_size', type=int, default=256)
    parser.add_argument('--mlp_hidden_size', type=int, default=1024)
    parser.add_argument('--img_channels', type=int, default=3)
    parser.add_argument('--pos_channels', type=int, default=4)
    parser.add_argument('--num_cross_heads', type=int, default=None)
    
    parser.add_argument('--dec_type',  type=str, default='transformer', help='type of decoder transformer or mlp')
    parser.add_argument('--cappa', type=float, default=-1)
    parser.add_argument('--mlp_dec_hidden',  type=int, default=2048, help='Dimension of decoder mlp hidden layers')
    parser.add_argument('--use_slot_proj',  type=bool_flag, default=True, help='Use an extra projection before MLP decoder')

    parser.add_argument('--which_encoder',  type=str, default='dino_vitb16', help='dino_vitb16, dino_vits8, dinov2_vitb14_reg, dinov2_vits14_reg, dinov2_vitb14, dinov2_vits14, mae_vitb16')
    parser.add_argument('--finetune_blocks_after',  type=int, default=8, help='finetune the blocks from this and after (counting from 0), for vit-b values greater than 12 means keep everything frozen')
    parser.add_argument('--encoder_final_norm',  type=bool_flag, default=False)
    parser.add_argument('--pretrained_encoder_weights', type=str, default=None)
    parser.add_argument('--use_second_encoder',  type= bool_flag, default = False, help='different encoder for input and target of decoder')

    
    parser.add_argument('--truncate',  type=str, default='bi-level', help='bi-level or fixed-point or none')
    parser.add_argument('--init_method', default='embedding', help='embedding or shared_gaussian')
    
    parser.add_argument('--train_permutations',  type=str, default='standard', help='which permutation')
    parser.add_argument('--eval_permutations',  type=str, default='standard', help='which permutation')

    parser.add_argument('--min-scale', type=float, default=0.08, help='minimum crop scale')
    parser.add_argument('--group-loss-weight', default=0.5, type=float, help='balancing weight of the grouping loss')
    parser.add_argument('--ctr-loss-weight', default=0.2, type=float, help='balancing weight of the grouping loss')
    parser.add_argument('--differ_loss_weight', default=0.5, type=float, help='balancing weight of the grouping loss')
    parser.add_argument('--kernel_size', default=1, type=int, help='projector_kernel')
    parser.add_argument('--top_k', default=7, type=int, help='decoder input slot num')

    parser.add_argument('--teacher-temp', default=0.07, type=float, help='teacher temperature')
    parser.add_argument('--student-temp', default=0.1, type=float, help='student temperature')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')
    parser.add_argument('--teacher-momentum', default=0.99, type=float, help='momentum value for the teacher model')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--center-momentum', default=0.9, type=float, help='momentum for the center')



    
    parser.add_argument('--ce_weight', type=float, default=5e-3, help='weight of the cross-entropy distilation loss')
    parser.add_argument('--final_ce_weight', type=float, default=None, help='final weight of the cross-entropy distilation loss')
    
    parser.add_argument('--teacher_checkpoint_path', help='teacher checkpoint')
    parser.add_argument('--teacher_free_smoke_test', type=bool_flag, default=False, help='Run without loading a teacher checkpoint by using student slots as a fallback')
    parser.add_argument('--teacher_truncate',  type=str, default = 'bi-level')
    parser.add_argument('--teacher_init_method',  type=str, default = 'mu_embedding')
    parser.add_argument('--teacher_train_permutations',  type=str, default='standard', help='which permutation')
    parser.add_argument('--teacher_eval_permutations',  type=str, default='standard', help='which permutation')
    parser.add_argument('--viz_resolution_factor', type=float, default=0.5)

    
    parser.add_argument('--device', type=str, default='auto', help='auto, cpu, cuda:0, or a CUDA index like 0')

    return parser

def train(args):
    print(datetime.today().isoformat())
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    if args.num_workers is None:
        args.num_workers = 0 if is_running_in_colab() else 4
    
    arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
    arg_str = '__'.join(arg_str_list)
    log_dir = os.path.join(args.log_path, datetime.today().isoformat())
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    writer.add_text('hparams', arg_str)
    
    if args.dataset == 'voc':
        train_dataset = PascalVOC(root=args.data_path, split='trainaug', image_size=args.image_size, mask_size = args.image_size)
        val_dataset = PascalVOC(root=args.data_path, split='val', image_size=args.val_image_size, mask_size = args.val_mask_size)
    elif args.dataset == 'coco':
        train_dataset = COCO2017(root=args.data_path, split='train', image_size=args.image_size, mask_size = args.image_size)
        val_dataset = COCO2017(root=args.data_path, split='val', image_size=args.val_image_size, mask_size = args.val_mask_size)
    elif args.dataset == 'movi':
        train_dataset = MOVi(root=os.path.join(args.data_path, 'train'), split='train', image_size=args.image_size, mask_size = args.image_size, frames_per_clip=9, predefined_json_paths = args.predefined_movi_json_paths)
        val_dataset = MOVi(root=os.path.join(args.data_path, 'validation'), split='validation', image_size=args.val_image_size, mask_size = args.val_mask_size)
    elif args.dataset == 'tumor':
        # Training: images-only (unsupervised). Validation: images + masks.
        # data_path/images/ → training images
        # val_data_path/images/ + val_data_path/masks/ → validation
        val_root = args.val_data_path if args.val_data_path else args.data_path
        train_dataset = TumorDataset(
            images_dir=os.path.join(args.data_path, 'images'),
            split='train',
            image_size=args.image_size,
            mask_size=args.image_size,
            mask_ext=args.mask_ext,
            mask_threshold=args.mask_threshold,
        )
        val_dataset = TumorDataset(
            images_dir=os.path.join(val_root, 'images'),
            masks_dir=os.path.join(val_root, 'masks'),
            split='val',
            image_size=args.val_image_size,
            mask_size=args.val_mask_size,
            mask_ext=args.mask_ext,
            mask_threshold=args.mask_threshold,
        )

    train_sampler = None
    val_sampler = None
    
    loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': device.type == 'cuda',
    }
    
    train_loader = DataLoader(train_dataset, sampler=train_sampler, shuffle=True, drop_last = True, batch_size=args.batch_size, **loader_kwargs)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, shuffle=False, drop_last = False, batch_size=args.eval_batch_size, **loader_kwargs)
    
    train_epoch_size = len(train_loader)
    val_epoch_size = len(val_loader)
    args.num_instances = len(val_loader.dataset)

    
    log_interval = train_epoch_size // 5
    
    if args.which_encoder == 'dino_vitb16':
        args.max_tokens = int((args.val_image_size/16)**2)
        encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        encoder_s = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    elif args.which_encoder == 'dino_vits8':
        args.max_tokens = int((args.val_image_size/8)**2)
        encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    elif args.which_encoder == 'dino_vitb8':
        args.max_tokens = int((args.val_image_size/8)**2)
        encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
    elif args.which_encoder == 'dinov2_vitb14':
        args.max_tokens = int((args.val_image_size/14)**2)
        encoder = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitb14')
        encoder_s = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitb14')
    elif args.which_encoder == 'dinov2_vits14':
        args.max_tokens = int((args.val_image_size/14)**2)
        encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    elif args.which_encoder == 'dinov2_vitb14_reg':
        args.max_tokens = int((args.val_image_size/14)**2)
        encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    elif args.which_encoder == 'dinov2_vits14_reg':
        args.max_tokens = int((args.val_image_size/14)**2)
        encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    elif args.which_encoder == 'mae_vitb16':
        args.max_tokens = int((args.val_image_size/16)**2)
        encoder = models_vit.__dict__["vit_base_patch16"](num_classes=0, global_pool=False, drop_path_rate=0)
        encoder_s = models_vit.__dict__["vit_base_patch16"](num_classes=0, global_pool=False, drop_path_rate=0)
        assert args.pretrained_encoder_weights is not None
        load_pretrained_encoder(encoder, args.pretrained_encoder_weights, prefix=None)      
        load_pretrained_encoder(encoder_s, args.pretrained_encoder_weights, prefix=None)      
    else:
        raise

    if 'encoder_s' not in locals():
        encoder_s = copy.deepcopy(encoder)
    
    encoder = encoder.eval()
    encoder_s = encoder_s.train()

    
    if args.use_second_encoder:
        encoder_second = copy.deepcopy(encoder).eval()
    else:
        encoder_second = None
    
    if args.num_cross_heads is None:
        args.num_cross_heads = args.num_heads
    

    student_model = SPOT(encoder, encoder_s,args, encoder_second)
    
    teacher_model = None
    if args.teacher_free_smoke_test:
        print('Running in teacher-free smoke test mode.')
    else:
        if not args.teacher_checkpoint_path:
            raise ValueError('Please provide --teacher_checkpoint_path, or set --teacher_free_smoke_test true for a smoke test.')

        args_teacher = copy.deepcopy(args)
        args_teacher.truncate = args.teacher_truncate
        args_teacher.init_method = args.teacher_init_method
        args_teacher.train_permutations = args.teacher_train_permutations
        args_teacher.eval_permutations = args.teacher_eval_permutations
        args_teacher.finetune_blocks_after = 100
        args_teacher.num_slots = 2

        teacher_model = Indicator(encoder, args_teacher)

        checkpoint = load_checkpoint(args.teacher_checkpoint_path, map_location='cpu')
        checkpoint['model'] = {k.replace("tf_dec.", "dec."): v for k, v in checkpoint['model'].items()} # compatibility with older runs
        teacher_model.load_state_dict(checkpoint['model'], strict = True)
        teacher_model = teacher_model.to(device).eval()

        for param in teacher_model.parameters():
            param.requires_grad = False  # not update by gradient
        # print(msg)

    if os.path.isfile(args.checkpoint_path):
        checkpoint = load_checkpoint(args.checkpoint_path, map_location='cpu')
        start_epoch = 0
        best_val_loss = math.inf
        best_epoch = 0
        best_val_ari = 0
        best_val_ari_slot = 0
        best_mbo_c = 0
        best_mbo_i = 0
        best_miou= 0 
        best_mbo_c_slot = 0
        best_mbo_i_slot = 0
        best_miou_slot= 0
        student_model.load_state_dict(checkpoint['model'], strict=False)
        msg = student_model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
    else:
        print('No checkpoint_path found')
        checkpoint = None
        start_epoch = 0
        best_val_loss = math.inf
        best_epoch = 0
        best_val_ari = 0
        best_val_ari_slot = 0
        best_mbo_c = 0
        best_mbo_i = 0
        best_miou= 0 
        best_mbo_c_slot = 0
        best_mbo_i_slot = 0
        best_miou_slot= 0
    

    student_model = student_model.to(device).train()
    
    lr_schedule = cosine_scheduler( base_value = args.lr_main,
                                    final_value = args.lr_min,
                                    epochs = args.epochs, 
                                    niter_per_ep = len(train_loader),
                                    warmup_epochs=int(args.lr_warmup_steps/(len(train_dataset)/args.batch_size)),
                                    start_warmup_value=0)

    if args.final_ce_weight == None:
        args.final_ce_weight = args.ce_weight

    ce_weight_schedule = cosine_scheduler( base_value = args.ce_weight,
                                final_value = args.final_ce_weight,
                                epochs = args.epochs, 
                                niter_per_ep = len(train_loader),
                                warmup_epochs=0,
                                start_warmup_value=0)
    
    optimizer = Adam([
        {'params': (param for name, param in student_model.named_parameters() if param.requires_grad), 'lr': args.lr_main},
    ])

    
    MBO_c_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).to(device)
    MBO_i_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).to(device)
    miou_metric = UnsupervisedMaskIoUMetric(matching="hungarian", ignore_background = True, ignore_overlaps = True).to(device)
    ari_metric = ARIMetric(foreground = True, ignore_overlaps = True).to(device)
    
    MBO_c_slot_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).to(device)
    MBO_i_slot_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).to(device)
    miou_slot_metric = UnsupervisedMaskIoUMetric(matching="hungarian", ignore_background = True, ignore_overlaps = True).to(device)
    ari_slot_metric = ARIMetric(foreground = True, ignore_overlaps = True).to(device)
    
    
    def get_teacher_targets(image):
        if teacher_model is not None:
            return teacher_model.forward_eval(image)

        emb_input_student = student_model.forward_encoder(image, student_model.encoder)
        slots_student, _, _, _, _ = student_model.slot_attn(emb_input_student)
        return emb_input_student.detach(), None, slots_student.detach(), None, None, None

    if teacher_model is not None:
        teacher_model.eval()

    for epoch in range(start_epoch, args.epochs):
        student_model.train()
    
        for batch, image in enumerate(train_loader):
            
            image = image.to(device)

            global_step = epoch * train_epoch_size + batch
    
            optimizer.param_groups[0]['lr'] = lr_schedule[global_step]
            lr_value = optimizer.param_groups[0]['lr']
            
            optimizer.zero_grad()
        
            with torch.no_grad():
                emb_input_teacher,default_slots_attns_teacher, slots_teacher, attn_logits_teacher,emb_input_mlp_teacher,property_mean_teacher = get_teacher_targets(image)
            
            
            loss_mse , loss_map,dec_slots_attns, _, _= student_model.forward_ours_stage2(image,emb_input_teacher,slots_teacher)

            ce_weight = ce_weight_schedule[global_step]

            total_loss = loss_mse + ce_weight *loss_map

            student_model.update_slot_attn_s(ema=True)

            total_loss.backward()
            clip_grad_norm_(student_model.parameters(), args.clip, 'inf')
            optimizer.step()

            with torch.no_grad():
                if batch % log_interval == 0:
                    print('Train Epoch: {:3} [{:5}/{:5}] \t lr = {:5} \t total_loss: {:F} \t loss_mse: {:F} \t loss_map: {:F}'.format(
                          epoch+1, batch, train_epoch_size, lr_value, total_loss.item(),loss_mse.item(),loss_map.item()))
                    writer.add_scalar('TRAIN/total_loss', total_loss.item(), global_step)
                    writer.add_scalar('TRAIN/lr_main', lr_value, global_step)
        with torch.no_grad():
            student_model.eval()

            val_mse = 0.
            counter = 0
    
            for batch, (image, true_mask_i, true_mask_c, mask_ignore) in enumerate(tqdm(val_loader)):
                image = image.to(device)
                true_mask_i = true_mask_i.to(device)
                true_mask_c = true_mask_c.to(device)
                mask_ignore = mask_ignore.to(device)
                
                batch_size = image.shape[0]
                counter += batch_size

                emb_input_teacher,default_slots_attns_teacher, slots_teacher, attn_logits_teacher,emb_input_mlp_teacher,property_mean_teacher = get_teacher_targets(image)

                mse,  dec_slots_attns, _, _ ,gamma,beta= student_model.forward_ours_eval(image,emb_input_teacher,slots_teacher)
                
                dec_attns = F.interpolate(dec_slots_attns, size=args.val_mask_size, mode='bilinear')

                dec_attns = dec_attns.unsqueeze(2) # shape [B, num_slots, 1, H, W]
    
                pred_dec_mask = dec_attns.argmax(1).squeeze(1)

    
                val_mse += mse.item()
                
                # Compute ARI, MBO_i and MBO_c, miou scores for both slot attention and decoder
                true_mask_i_reshaped = torch.nn.functional.one_hot(true_mask_i).to(torch.float32).permute(0,3,1,2).to(device)
                true_mask_c_reshaped = torch.nn.functional.one_hot(true_mask_c).to(torch.float32).permute(0,3,1,2).to(device)
                pred_dec_mask_reshaped = torch.nn.functional.one_hot(pred_dec_mask).to(torch.float32).permute(0,3,1,2).to(device)
                
                MBO_i_metric.update(pred_dec_mask_reshaped, true_mask_i_reshaped, mask_ignore)
                MBO_c_metric.update(pred_dec_mask_reshaped, true_mask_c_reshaped, mask_ignore)
                miou_metric.update(pred_dec_mask_reshaped, true_mask_i_reshaped, mask_ignore)
                ari_metric.update(pred_dec_mask_reshaped, true_mask_i_reshaped, mask_ignore)
            

            gamma_diff = torch.abs(gamma - 1).mean()
            beta_diff = torch.abs(beta - 0).mean()
        
            print("gamma_diff:",gamma_diff)
            print("beta_diff:",beta_diff)
            student_model.compare_param()

            val_mse /= (val_epoch_size)
            ari = 100 * ari_metric.compute()
            mbo_c = 100 * MBO_c_metric.compute()
            mbo_i = 100 * MBO_i_metric.compute()
            miou = 100 * miou_metric.compute()
            val_loss = val_mse
            writer.add_scalar('VAL/mse', val_mse, epoch+1)
            writer.add_scalar('VAL/ari (decoder)', ari, epoch+1)
            writer.add_scalar('VAL/mbo_c', mbo_c, epoch+1)
            writer.add_scalar('VAL/mbo_i', mbo_i, epoch+1)
            writer.add_scalar('VAL/miou', miou, epoch+1)
            
            print(args.log_path)
            print('====> Epoch: {:3} \t Loss = {:F} \t MSE = {:F} \t ARI = {:F} \t  mBO_c = {:F} \t mBO_i = {:F} \t miou = {:F} \t '.format(
                epoch+1, val_loss, val_mse, ari, mbo_c, mbo_i, miou))
            
            ari_metric.reset()
            MBO_c_metric.reset()
            MBO_i_metric.reset()
            miou_metric.reset()
            MBO_c_slot_metric.reset()
            MBO_i_slot_metric.reset()
            ari_slot_metric.reset()
            miou_slot_metric.reset()
            
            if  (best_mbo_c < mbo_c) or (best_mbo_i < mbo_i):
                best_val_loss = val_loss
                best_val_ari = ari
                best_mbo_c = mbo_c
                best_mbo_i = mbo_i
                best_miou = miou
                best_epoch = epoch + 1
    
                torch.save(student_model.state_dict(), os.path.join(log_dir, 'best_model.pt'))

    
            writer.add_scalar('VAL/best_loss', best_val_loss, epoch+1)
    
            checkpoint = {
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'best_val_ari': best_val_ari,
                'best_val_ari_slot': best_val_ari_slot,
                'best_mbo_c':best_mbo_c,
                'best_mbo_i':best_mbo_i,
                'best_miou':best_miou,
                'best_mbo_c_slot':best_mbo_c_slot,
                'best_mbo_i_slot':best_mbo_i_slot,
                'best_miou_slot':best_miou_slot,
                'best_epoch': best_epoch,
                'model': student_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
    
            torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pt.tar'))
    
            print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SPOT (2)', parents=[get_args_parser()])
    args = parser.parse_args()
    train(args)
