import sys
import os
import glob
import torch
import torchio as tio
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from ddpm.BiFlowNet import GaussianDiffusion, BiFlowNet
from AutoEncoder.model.PatchVolume import patchvolumeAE

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_dir)
sys.path.append(project_root)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 1. 配置路径 (指向你的微调权重)
    # ==========================================
    AE_CKPT_PATH = r"E:\deep learning code\3D_MedDiffusion_Code\PatchVolume4x_s2.ckpt"
    
    # 🚨 用户要求使用 8800 步的权重 (Loss 波动较小)
    LDM_CKPT_PATH = glob.glob(r"E:\deep learning code\3D_MedDiffusion_Code\results_rhuh_ft\rhuh_ft_*\checkpoints\0008800.pt")[-1]

    # 🚨 输入文件夹：标准化后的 RHUH 原图
    INPUT_DIR = r"E:\deep learning code\3D_MedDiffusion_Code\RHUH_standardized"
    
    # 🚨 原始数据文件夹 (用于提取坐标元数据进行还原)
    RAW_DATA_ROOT = r"E:\deep learning code\3D_MedDiffusion_Code\RHUH-GBM_nii_v1 - 副本"
    REF_TEMPL_PATH = r"E:\deep learning code\3D_MedDiffusion_Code\BraTsdataset\data\BraTS-GLI-00005-100-t1c.nii"

    # 🚨 输出文件夹
    OUTPUT_DIR = r"E:\deep learning code\3D_MedDiffusion_Code\RHUH_FT_predictions_RESTORED"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Helper for Safe Reading
    import tempfile
    import shutil
    def safe_read(path):
        suffix = "".join(Path(path).suffixes) if hasattr(Path(path), 'suffixes') else ".nii.gz"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            shutil.copy(path, tf.name)
            img = sitk.ReadImage(tf.name)
        os.remove(tf.name)
        return img

    from pathlib import Path
    template_img = safe_read(REF_TEMPL_PATH)

    # ==========================================
    # 2. 初始化网络
    # ==========================================
    print(f"🚀 Loading RHUH Fine-tuned weights: {os.path.basename(LDM_CKPT_PATH)}")

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
    # 3. 批量循环处理 (只针对测试集)
    # ==========================================
    import json
    TEST_JSON = r"E:\deep learning code\3D_MedDiffusion_Code\rhuh_test_pairs.json"
    if not os.path.exists(TEST_JSON):
        # Fallback if specific split doesn't exist yet
        file_list = glob.glob(os.path.join(INPUT_DIR, "*-100-*.nii.gz"))
    else:
        with open(TEST_JSON, 'r') as f:
            test_data = json.load(f)
        file_list = [item["pre_op_img"] for item in test_data]
        test_ids = [item["p_id"] for item in test_data]

    print(f"📦 准备对 {len(file_list)} 个测试用例进行生成并还原坐标...")

    for idx, file_path in enumerate(file_list):
        filename = os.path.basename(file_path)
        if 'test_ids' in locals():
            p_id = test_ids[idx]
        else:
            p_id = filename.split("-100-")[0]
        out_name = filename.replace('.nii.gz', '') + "_RESTORED.nii.gz"
        save_path = os.path.join(OUTPUT_DIR, out_name)

        print(f"\n▶️ [{idx + 1}/{len(file_list)}] 处理中: {p_id}")

        # 1. 加载并编码 (Latent Extraction)
        from monai.transforms import Compose, LoadImaged, ScaleIntensityd, EnsureChannelFirstd, Resized
        keys = ["img"]
        transform = Compose([
            LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys),
            Resized(keys=keys, spatial_size=(96, 192, 96), mode="trilinear"), 
            ScaleIntensityd(keys=keys, minv=-1.0, maxv=1.0)
        ])
        raw_data = transform({"img": file_path})["img"]
        # 🛡️ 保持原汁原味：删掉转置，保住最高清晰度
        raw_data = raw_data.unsqueeze(0).to(device) 
        
        with torch.no_grad():
            pre_latent_raw = AE.patch_encode(raw_data, patch_size=96)
            pre_latent = ((pre_latent_raw - c_min) / (c_max - c_min)) * 2.0 - 1.0

        # ==========================================
        # 2. Diffusion 采样准备
        # ==========================================
        volume_size = [24, 48, 24]
        if pre_latent.shape[-3:] != tuple(volume_size):
            pre_latent_cpu = pre_latent.squeeze(0).cpu()
            pre_latent = tio.Resize(volume_size)(tio.ScalarImage(tensor=pre_latent_cpu)).data.unsqueeze(0).to(device)

        z = torch.randn(1, 8, volume_size[0], volume_size[1], volume_size[2], device=device)
        y = torch.tensor([0], device=device)
        res = torch.tensor(volume_size, device=device) / 64.0

        with torch.no_grad():
            predicted_latent = diffusion.p_sample_loop(model, z, y=y, res=res, hint=pre_latent)
            torch.cuda.empty_cache()

            # 3. 解码回空间特征 (96, 192, 96)
            predicted_unscaled = (((predicted_latent + 1.0) / 2.0) * (c_max - c_min)) + c_min
            test_volume = AE.decode(predicted_unscaled, quantize=False)
            test_volume = (test_volume + 1.0) / 2.0
            test_volume = test_volume.clamp(0.0, 1.0) # [1, 1, 96, 192, 96]
            
            # --- 🛠️ 质量与方向的终极平衡 ---
            # 1. 使用 GPU 插值提升锐利度 (恢复到 182x218x182 的标准参考网格)
            # 这步是保住“高清晰度”的关键逻辑
            test_volume_high = F.interpolate(test_volume, size=(182, 218, 182), mode='trilinear', align_corners=False)
            
            # 2. 轴向对齐
            # 模型输出是 [D, H, W]，为了 SITK 读到 [W, H, D] 正视图
            pred_array = test_volume_high.squeeze(0).squeeze(0).cpu().numpy()
            pred_sitk_std = sitk.GetImageFromArray(pred_array.transpose(2, 1, 0))
            pred_sitk_std.CopyInformation(template_img)

            # 2. 找到原始文件进行采样
            orig_parent = Path(RAW_DATA_ROOT) / p_id / "0"
            orig_candidates = list(orig_parent.glob("*_t1ce.nii.gz"))
            if not orig_candidates:
                print(f"⚠️ 找不到原始文件进行坐标还原: {p_id}")
                continue
            orig_sitk = safe_read(str(orig_candidates[0]))
            
            # 3. 重新计算对齐变换
            align_transform = sitk.CenteredTransformInitializer(
                template_img, orig_sitk, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            
            # 4. 执行单阶段重采样 (Low-Res -> Original-High-Res)
            # 这步会直接把 96x192x96 还原到 240x240x155，避免中间环节的像素模糊
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(orig_sitk)
            resampler.SetInterpolator(sitk.sitkBSpline) # 使用 BSpline 获得更平滑锐利的边缘
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(align_transform.GetInverse())
            
            final_restored_sitk = resampler.Execute(pred_sitk_std)
            
            # 保存
            # 🛠️ 修复 Windows 权限问题：先关闭句柄，再写入并移动
            temp_fd, temp_name = tempfile.mkstemp(suffix=".nii.gz")
            os.close(temp_fd) # 立即关闭句柄，释放文件锁
            try:
                sitk.WriteImage(final_restored_sitk, temp_name)
                if os.path.exists(save_path):
                    os.remove(save_path)
                shutil.move(temp_name, save_path)
                print(f"✅ 坐标还原成功: {out_name}")
            except Exception as e:
                print(f"❌ 保存失败: {str(e)}")
                if os.path.exists(temp_name): os.remove(temp_name)

    print("\n🎉 RHUH 坐标原位还原推理全部完成！请查看 RHUH_FT_predictions_RESTORED 文件夹。")

    print("\n🎉 RHUH 微调后推理全部完成！请查看 RHUH_FT_predictions 文件夹。")

if __name__ == "__main__":
    main()
