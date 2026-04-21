import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from collections import OrderedDict
from glob import glob
from time import time
import argparse
import logging
from ddpm.BiFlowNet import GaussianDiffusion, BiFlowNet
from AutoEncoder.model.PatchVolume import patchvolumeAE
import torchio as tio
import copy
from torch.cuda.amp import autocast, GradScaler
from dataset.Singleres_dataset import Singleres_dataset
from torch.utils.data import DataLoader

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def create_logger(logging_dir):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = old_weight * self.beta + (1 - self.beta) * up_weight

def main(args):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Setup Experiment Dir
    os.makedirs(args.results_dir, exist_ok=True)
    experiment_dir = f"{args.results_dir}/rhuh_ft_{int(time())}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    samples_dir = f"{experiment_dir}/samples"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Fine-tuning session started at {experiment_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 🧠 Load AE
    ae = patchvolumeAE.load_from_checkpoint(args.AE_ckpt).to(device)
    ae.eval()

    # 🦾 Initialize Model
    model = BiFlowNet(
        dim=72, dim_mults=[1, 1, 2, 4, 8], channels=8, init_kernel_size=3,
        cond_classes=7, use_sparse_linear_attn=[0, 0, 0, 1, 1],
        vq_size=64, num_mid_DiT=1, patch_size=2, sub_volume_size=[24, 48, 24]
    ).to(device)
    
    ema_model = copy.deepcopy(model).to(device)
    ema_helper = EMA(0.999)
    
    diffusion = GaussianDiffusion(channels=8, timesteps=1000, loss_type='l1').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    scaler = GradScaler()

    # 🚀 Load Weights (Resume or Pre-trained)
    start_epoch = 0
    train_steps = 0
    if args.resume:
        logger.info(f"Resuming RHUH training from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        ema_model.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['opt'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint.get('epoch', 0)
        train_steps = int(os.path.basename(args.resume).split('.')[0])
        logger.info(f"Resume successful! Starting from step {train_steps}, epoch {start_epoch}")
    elif args.pretrained_ckpt:
        logger.info(f"Loading pre-trained BraTS weights: {args.pretrained_ckpt}")
        checkpoint = torch.load(args.pretrained_ckpt, map_location='cpu', weights_only=False)
        state_dict = checkpoint['ema']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        ema_model.load_state_dict(state_dict, strict=True)
        logger.info("Successfully initialized model with BraTS knowledge.")

    # Dataset
    dataset = Singleres_dataset(args.data_path, resolution=[24, 48, 24])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Fixed Val Sample for RHUH
    val_post, val_pre, val_res = dataset[0]
    val_pre_fixed = val_pre.unsqueeze(0).to(device)
    val_res_fixed = val_res.unsqueeze(0).to(device)
    val_y_fixed = torch.zeros(1, dtype=torch.long, device=device)

    model.train()
    logger.info(f"Beginning RHUH Fine-tuning for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        for z_post, z_pre, res in loader:
            z_post, z_pre, res = z_post.to(device), z_pre.to(device), res.to(device)
            y = torch.zeros(z_post.shape[0], dtype=torch.long, device=device)

            optimizer.zero_grad()
            with autocast():
                t = torch.randint(0, 1000, (z_post.shape[0],), device=device)
                loss = diffusion.p_losses(model, x_start=z_post, t=t, y=y, res=res, hint=z_pre)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            ema_helper.update_model_average(ema_model, model)
            train_steps += 1

            if train_steps % 50 == 0:
                logger.info(f"Epoch {epoch} | Step {train_steps} | Loss: {loss.item():.4f}")

            if train_steps % args.ckpt_every == 0:
                # Save Checkpoint
                ckpt_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "scaler": scaler.state_dict(),
                    "opt": optimizer.state_dict()
                }, ckpt_path)
                logger.info(f"Saved RHUH checkpoint: {ckpt_path}")

                # Save Sample
                with torch.no_grad():
                    z = torch.randn(1, 8, 24, 48, 24, device=device)
                    samples = diffusion.p_sample_loop(ema_model, z, y=val_y_fixed, res=val_res_fixed, hint=val_pre_fixed)
                    
                    c_min, c_max = ae.codebook.embeddings.min(), ae.codebook.embeddings.max()
                    samples = ((samples + 1.0) / 2.0) * (c_max - c_min) + c_min
                    volume = ae.decode(samples, quantize=False)
                    
                    v_path = f"{samples_dir}/{train_steps:07d}.nii.gz"
                    # Simple affine for viz
                    affine = np.diag([1, 1, 1, 1.0])
                    tio.ScalarImage(tensor=volume[0].cpu().permute(0, 3, 2, 1), affine=affine).save(v_path)
                    logger.info(f"Validation sample saved to {v_path}")

    logger.info("RHUH Fine-tuning complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=r"E:\deep learning code\3D_MedDiffusion_Code\rhuh_train_pairs.json")
    parser.add_argument("--pretrained-ckpt", type=str, default=r"E:\deep learning code\3D_MedDiffusion_Code\results\biflownet\000-biflownet_v1\checkpoints\0098500.pt")
    parser.add_argument("--resume", type=str, default=None, help="指向 RHUH 自己的 .pt 权重文件以实现续训")
    parser.add_argument("--AE-ckpt", type=str, default=r"E:\deep learning code\3D_MedDiffusion_Code\PatchVolume4x_s2.ckpt")
    parser.add_argument("--results-dir", type=str, default=r"E:\deep learning code\3D_MedDiffusion_Code\results_rhuh_ft")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=200)
    args = parser.parse_args()
    main(args)
