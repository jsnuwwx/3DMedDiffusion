import sys
import os
import glob
import torch
import torchio as tio
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from ddpm.BiFlowNet import GaussianDiffusion, BiFlowNet
from AutoEncoder.model.PatchVolume import patchvolumeAE

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_dir)
sys.path.append(project_root)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 1. 批量配置路径 (改成文件夹)
    # ==========================================
    AE_CKPT_PATH = r"E:\deep learning code\3D_MedDiffusion_Code\PatchVolume4x_s2.ckpt"
    LDM_CKPT_PATH = r"E:\deep learning code\3D_MedDiffusion_Code\results\biflownet\000-biflownet_v1\checkpoints\0098500.pt"

    # 🚨 输入文件夹：指向你存放所有病人特征图的目录
    INPUT_DIR = r"E:\deep learning code\3D_MedDiffusion_Code\RHUH_standardized"
    # 🚨 输出文件夹：批量生成的结果都会保存在这里
    OUTPUT_DIR = r"./batch_predictions_RHUH"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # 2. 核心网络初始化 (只加载一次！)
    # ==========================================
    print("🚀 Loading BiFlowNet and AE into memory...")

    model = BiFlowNet(
        dim=72, dim_mults=[1, 1, 2, 4, 8], channels=8, init_kernel_size=3,
        cond_classes=7, use_sparse_linear_attn=[0, 0, 0, 1, 1],
        vq_size=64, num_mid_DiT=1, patch_size=2, sub_volume_size=[24, 48, 24]
    ).to(device)

    diffusion = GaussianDiffusion(channels=8, timesteps=1000, loss_type='l1').to(device)

    checkpoint = torch.load(LDM_CKPT_PATH, map_location=device, weights_only=False)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['ema'].items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    AE = patchvolumeAE.load_from_checkpoint(AE_CKPT_PATH).to(device)
    AE.eval()

    c_min = AE.codebook.embeddings.min()
    c_max = AE.codebook.embeddings.max()

    # ==========================================
    # 3. 开始批量循环处理
    # ==========================================
    # 自动搜索文件夹下所有的 .nii 和 .nii.gz 文件
    file_list = glob.glob(os.path.join(INPUT_DIR, "*-100-*.nii")) + glob.glob(os.path.join(INPUT_DIR, "*-100-*.nii.gz"))

    if not file_list:
        print(f"❌ 没有在 {INPUT_DIR} 找到任何文件，请检查路径！")
        return

    print(f"📦 扫描完毕，共发现 {len(file_list)} 个病例，开启批量流水线！\n" + "=" * 50)

    for idx, file_path in enumerate(file_list):
        filename = os.path.basename(file_path)
        # 自动生成对应的输出文件名，加上 _predicted 后缀
        out_name = filename.replace('.nii.gz', '').replace('.nii', '') + "_predicted.nii.gz"
        save_path = os.path.join(OUTPUT_DIR, out_name)

        print(f"\n▶️ [{idx + 1}/{len(file_list)}] 正在处理: {filename}")

        if file_path.endswith('.gz'):
            from monai.transforms import Compose, LoadImaged, ScaleIntensityd, EnsureChannelFirstd, Resized
            keys = ["img"]
            transform = Compose([
                LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys),
                Resized(keys=keys, spatial_size=(96, 96, 96), mode="trilinear"),
                ScaleIntensityd(keys=keys, minv=-1.0, maxv=1.0)
            ])
            raw_data = transform({"img": file_path})["img"]
            raw_data = raw_data.transpose(1, 3).transpose(2, 3).unsqueeze(0).to(device)
            with torch.no_grad():
                pre_latent_raw = AE.patch_encode(raw_data, patch_size=96)
                pre_latent = ((pre_latent_raw - c_min) / (c_max - c_min)) * 2.0 - 1.0
        else:
            pre_latent = tio.ScalarImage(file_path).data.to(torch.float32).unsqueeze(0).to(device)

        volume_size = [24, 48, 24]
        if pre_latent.shape[-3:] != tuple(volume_size):
            # 🧠 Fix: tio.Resize requires the tensor to be on CPU to initialize as a ScalarImage
            pre_latent_cpu = pre_latent.squeeze(0).cpu()
            pre_latent = tio.Resize(volume_size)(tio.ScalarImage(tensor=pre_latent_cpu)).data.unsqueeze(0).to(device)

        z = torch.randn(1, 8, volume_size[0], volume_size[1], volume_size[2], device=device)
        y = torch.tensor([0], device=device)
        res = torch.tensor(volume_size, device=device) / 64.0

        with torch.no_grad():
            # 推演术后特征
            predicted_latent = diffusion.p_sample_loop(model, z, y=y, res=res, hint=pre_latent)

            # 注意：这里删除了 del model，只清理显存缓存
            torch.cuda.empty_cache()

            # 解码与拉伸
            predicted_unscaled = (((predicted_latent + 1.0) / 2.0) * (c_max - c_min)) + c_min
            test_volume = AE.decode(predicted_unscaled, quantize=False)
            test_volume = test_volume.detach().squeeze(0).cpu()
            test_volume = (test_volume + 1.0) / 2.0
            test_volume = test_volume.clamp(0.0, 1.0)

            target_shape = (182, 218, 182)
            test_volume_5d = test_volume.unsqueeze(0)
            restored_volume = F.interpolate(test_volume_5d, size=target_shape, mode='trilinear', align_corners=False)

            final_array = restored_volume.squeeze(0).squeeze(0).cpu().numpy()

            # 翻转修复坐标
            final_array = np.flip(final_array, axis=2)
            final_array = np.flip(final_array, axis=1)
            final_array = final_array.copy()

            # MNI152 标准仿射矩阵
            affine = np.array([
                [1.0, 0.0, 0.0, -90.0],
                [0.0, -1.0, 0.0, 126.0],
                [0.0, 0.0, -1.0, -72.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float32)

            nifti_img = nib.Nifti1Image(final_array, affine)
            nib.save(nifti_img, save_path)

            print(f"✅ 生成成功并保存至: {out_name}")

    print("\n🎉 全部任务处理完成！请前往 batch_predictions_PERFECT 文件夹查看。")


if __name__ == "__main__":
    main()