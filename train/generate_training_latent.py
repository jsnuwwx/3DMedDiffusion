import sys
import os
import torch
import torchio as tio
import argparse
from torch.utils.data import DataLoader
from os.path import join

# 确保能找到你的模型定义
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from AutoEncoder.model.PatchVolume import patchvolumeAE
from dataset.Singleres_dataset import Singleres_dataset


def generate(args):
    print(f"Initializing dataset: {args.data_path}")
    # 自动识别数据集（如果你的类支持读取目录）
    tr_dataset = Singleres_dataset(root_dir=args.data_path, resolution=args.resolution, generate_latents=True)

    # num_workers=0 是为了彻底避开 Linux 的 Bus error
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    print(f"Loading AE checkpoint: {os.path.basename(args.AE_ckpt)}")
    AE = patchvolumeAE.load_from_checkpoint(args.AE_ckpt)
    AE = AE.to(device)
    AE.eval()

    print(f"Starting extraction, total {len(tr_dataset)} files...")

    with torch.no_grad():
        for step, (sample, file_path) in enumerate(tr_dataloader):
            # Singleres_dataset 现在返回的是 [-1, 1] 的 Image 张量 (96, 96, 96)
            sample = sample.to(device)

            # 使用 VQ-GAN 的 Encoder 进行压缩
            # z 的范围通常取决于 codebook 的 embeddings 范围
            z = AE.patch_encode(sample, patch_size=96)

            # 🛠️ 核心修正：将 Latent 映射到 Diffusion 喜欢的 [-1, 1] 空间
            # 注意：这里的 c_min/c_max 是训练好的 AE 的特性，必须在这里对齐
            c_min = AE.codebook.embeddings.min()
            c_max = AE.codebook.embeddings.max()
            
            # 先反归一化到 codebook 的原始范围（如果需要），或者直接按比例缩放
            # 这里我们遵循原作者逻辑，将 z 映射到 [-1, 1]
            output = ((z - c_min) / (c_max - c_min)) * 2.0 - 1.0

            output = output.cpu()

            for output_, path in zip(output, file_path):
                # 自动构建保存路径，例如 data -> data_latents
                dir_name = os.path.basename(os.path.dirname(path))
                latent_dir_name = dir_name + '_latents'
                new_save_path = path.replace(dir_name, latent_dir_name)

                os.makedirs(os.path.dirname(new_save_path), exist_ok=True)

                # 使用 TorchIO 保存为 .nii 文件
                img = tio.ScalarImage(tensor=output_)
                img.save(new_save_path)
                print(f"Saved: {os.path.basename(new_save_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="指向你的 .nii 原图目录")
    parser.add_argument("--AE-ckpt", type=str, required=True, help="巅峰权重 .ckpt 文件的路径")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument('--resolution', nargs='+', type=int, default=[96, 96, 96], help="提取特征前对原图进行的 Resize 尺寸 (例如 96 192 96)")
    args = parser.parse_args()
    generate(args)
