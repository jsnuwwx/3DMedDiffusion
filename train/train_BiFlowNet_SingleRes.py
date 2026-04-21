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
from ddpm.BiFlowNet import GaussianDiffusion
from ddpm.BiFlowNet import BiFlowNet
from AutoEncoder.model.PatchVolume import patchvolumeAE
import torchio as tio
import copy
from torch.cuda.amp import autocast, GradScaler
import random
from dataset.Singleres_dataset import Singleres_dataset
from torch.utils.data import DataLoader

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

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

def _ddp_dict(_dict):
    new_dict = {}
    for k in _dict:
        new_dict['module.' + k] = _dict[k]
    return new_dict

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def main(args):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    start_epoch = 0
    train_steps = 0
    
    # 📡 Setup DDP & Device (Windows / Single-GPU Bypass)
    if 'WORLD_SIZE' in os.environ or 'RANK' in os.environ:
        backend = "gloo" if os.name == 'nt' else "nccl"
        dist.init_process_group(backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        print("Running in Single-GPU mode (DDP bypassed for Windows stability).")

    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}, device={device}")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        if args.ckpt is None:
            experiment_index = len(glob(f"{args.results_dir}/*"))
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{args.name}"
        else:
            experiment_dir = os.path.dirname(os.path.dirname(args.ckpt))
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        samples_dir = f"{experiment_dir}/samples"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # 🧠 Load AE
    ae = patchvolumeAE.load_from_checkpoint(args.AE_ckpt).to(device)
    ae.eval()

    # 🦾 Initialize Model
    model = BiFlowNet(
        dim=args.model_dim,
        dim_mults=args.dim_mults,
        channels=args.volume_channels,
        cond_classes=args.num_classes,
        use_sparse_linear_attn=args.use_attn,
        vq_size=args.vq_size,
        num_mid_DiT=args.num_dit,
        patch_size=args.patch_size,
        sub_volume_size=args.resolution
    ).to(device)
    
    ema_model = copy.deepcopy(model).to(device)
    ema = EMA(0.995)
    
    if dist.is_initialized():
        model = DDP(model, device_ids=[device])
    
    diffusion = GaussianDiffusion(
        channels=args.volume_channels,
        timesteps=args.timesteps,
        loss_type=args.loss_type,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4) # Standard AdamW
    scaler = GradScaler(enabled=args.enable_amp)
    amp = args.enable_amp

    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'), weights_only=False)
        # ✅ Handle potential DDP wrapping in checkpoint
        state_dict = checkpoint['model']
        if not dist.is_initialized() and any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        ema_model.load_state_dict(checkpoint['ema'], strict=True)
        scaler.load_state_dict(checkpoint['scaler'])
        opt.load_state_dict(checkpoint['opt'])
        start_epoch = checkpoint.get('epoch', 0)
        train_steps = int(os.path.basename(args.ckpt).split('.')[0])
        logger.info(f'Loaded checkpoint: {args.ckpt} at step {train_steps}')

    # Dataset
    dataset = Singleres_dataset(args.data_path, resolution=args.resolution)
    loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, pin_memory=True
    )

    # Fixed validation sample: always use the first patient for consistent progress tracking
    val_post, val_pre, val_res = dataset[0]
    val_pre_fixed = val_pre.unsqueeze(0).to(device)
    val_res_fixed = val_res.unsqueeze(0).to(device)
    val_y_fixed = torch.zeros(1, dtype=torch.long, device=device)

    model.train()
    ema_model.eval()
    
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Beginning epoch {epoch}...")

        for z_post, z_pre, res in loader:
            b = z_post.shape[0]
            z_post = z_post.to(device)
            z_pre = z_pre.to(device)
            res = res.to(device)
            y = torch.zeros(b, dtype=torch.long, device=device) # Unified virtual label

            opt.zero_grad()
            with autocast(enabled=amp):
                t = torch.randint(0, diffusion.num_timesteps, (b,), device=device)
                loss = diffusion.p_losses(model, x_start=z_post, t=t, y=y, res=res, hint=z_pre)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % 10 == 0:
                if train_steps < 2000:
                    source_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    ema_model.load_state_dict(source_dict, strict=True)
                else:
                    ema.update_model_average(ema_model, model)

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                if dist.is_initialized():
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / world_size
                else:
                    avg_loss = avg_loss.item()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    model_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    checkpoint = {
                        "model": model_dict, "ema": ema_model.state_dict(),
                        "scaler": scaler.state_dict(), "opt": opt.state_dict(),
                        "args": args, "epoch": epoch
                    }
                    ckpt_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, ckpt_path)
                    logger.info(f"Saved checkpoint to {ckpt_path}")

                    # Keep only last 3 checkpoints to save disk space
                    all_ckpts = sorted(glob(f"{checkpoint_dir}/*.pt"))
                    if len(all_ckpts) > 3:
                        for old_ckpt in all_ckpts[:-3]:
                            os.remove(old_ckpt)
                            logger.info(f"Deleted old checkpoint {os.path.basename(old_ckpt)} to save space")

                    # Sampling Validation
                    logger.info("Running sampling validation...")
                    with torch.no_grad():
                        z = torch.randn(1, args.volume_channels, args.resolution[0], args.resolution[1], args.resolution[2], device=device)
                        # Use fixed validation patient (not the current training batch) for consistent tracking
                        samples = diffusion.p_sample_loop(ema_model, z, y=val_y_fixed, res=val_res_fixed, hint=val_pre_fixed)
                        
                        # De-normalize to codebook space
                        c_min = ae.codebook.embeddings.min()
                        c_max = ae.codebook.embeddings.max()
                        samples = ((samples + 1.0) / 2.0) * (c_max - c_min) + c_min
                        
                        volume = ae.decode(samples, quantize=False)
                        v_path = f"{samples_dir}/{train_steps:07d}.nii.gz"
                        sx = 180.0 / (args.resolution[2] * 4)
                        sy = 220.0 / (args.resolution[1] * 4)
                        sz = 145.0 / (args.resolution[0] * 4)
                        affine = np.diag([sx, sy, sz, 1.0])
                        vol_tio = volume[0].cpu().permute(0, 3, 2, 1)
                        tio.ScalarImage(tensor=vol_tio, affine=affine).save(v_path)
                        logger.info(f"Sample saved to {v_path}")

                if dist.is_initialized():
                    dist.barrier()
                torch.cuda.empty_cache()

    logger.info("Done!")
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--AE-ckpt", type=str, required=True)
    parser.add_argument("--name", type=str, default="BiFlowNet")
    parser.add_argument("--volume-channels", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--loss-type", type=str, default='l1')
    parser.add_argument("--model-dim", type=int, default=72)
    parser.add_argument("--dim-mults", nargs='+', type=int, default=[1, 1, 2, 4, 8])
    parser.add_argument("--use-attn", nargs='+', type=int, default=[0, 0, 0, 1, 1])
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--num-dit", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--enable_amp", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument('--resolution', nargs='+', type=int, default=[24, 48, 24])
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--vq-size", type=int, default=64)
    args = parser.parse_args()
    main(args)
