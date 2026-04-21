import sys
import os
import torch
import torchio as tio
import argparse
import json
from AutoEncoder.model.PatchVolume import patchvolumeAE


def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 载入巅峰 AE 权重...")
    AE = patchvolumeAE.load_from_checkpoint(args.AE_ckpt).to(device)
    AE.eval()

    with open(args.data_path, 'r') as f:
        pairs = json.load(f)

    print(f"⚡ 开始提取 {len(pairs)} 对（术前+术后）文件...")

    with torch.no_grad():
        for step, item in enumerate(pairs):
            for key in ["pre_op_img", "post_op_img"]:
                path = item[key]
                if not os.path.exists(path):
                    continue

                # 1. 读取与归一化
                img = tio.ScalarImage(path)
                imageout = img.data.to(torch.float32) * 2.0 - 1.0

                # 2. 🚨 致命修复：添加原作者的神秘旋转！确保 AE 认识！
                imageout = imageout.transpose(1, 3).transpose(2, 3)
                imageout = imageout.unsqueeze(0).to(device)

                # 3. 编码
                z = AE.patch_encode(imageout, patch_size=96)

                # 4. 缩放公式
                c_min = AE.codebook.embeddings.min()
                c_max = AE.codebook.embeddings.max()
                output = ((z - c_min) / (c_max - c_min)) * 2.0 - 1.0

                # 5. 保存
                output = output.squeeze(0).cpu()
                dir_name = os.path.basename(os.path.dirname(path))
                latent_dir_name = dir_name + '_latents'
                new_save_path = path.replace(dir_name, latent_dir_name)
                os.makedirs(os.path.dirname(new_save_path), exist_ok=True)
                tio.ScalarImage(tensor=output).save(new_save_path)

            print(f"[{step + 1}/{len(pairs)}] ✅ 完成: {os.path.basename(item['pre_op_img'])} 及其对应术后特征")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--AE-ckpt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    generate(args)