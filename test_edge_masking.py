import os
import SimpleITK as sitk
from pathlib import Path
import tempfile
import shutil
import numpy as np

def safe_read(path):
    suffix = "".join(Path(path).suffixes) if hasattr(Path(path), 'suffixes') else ".nii.gz"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        shutil.copy(path, tf.name)
        img = sitk.ReadImage(tf.name)
    os.remove(tf.name)
    return img

def main():
    print("Starting Soft-Masked Edge-Guided pipeline test...")
    
    # 路径配置
    REF_TEMPL_PATH = r"E:\deep learning code\3D_MedDiffusion_Code\BraTsdataset\data\BraTS-GLI-00005-100-t1c.nii"
    RAW_DATA_ROOT = Path(r"E:\deep learning code\3D_MedDiffusion_Code\RHUH-GBM_nii_v1 - 副本")
    OUTPUT_DIR = Path(r"E:\deep learning code\3D_MedDiffusion_Code\RHUH_Masking_Test")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    target_id = "RHUH-0010"
    print(f"Target case: {target_id}")
    
    pre_op_path = RAW_DATA_ROOT / target_id / "0" / f"{target_id}_0_t1ce.nii.gz"
    mask_path = RAW_DATA_ROOT / target_id / "0" / f"{target_id}_0_segmentations.nii.gz"
    
    if not pre_op_path.exists() or not mask_path.exists():
        print("Error: Could not find pre_op or mask files.")
        return
        
    print("Step 1: Loading data and aligning to MNI152 template...")
    template_img = safe_read(REF_TEMPL_PATH)
    pre_img = safe_read(str(pre_op_path))
    mask_img = safe_read(str(mask_path))
    
    # 计算空间对齐 (Center Alignment)
    align_transform = sitk.CenteredTransformInitializer(
        template_img, pre_img, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    # 对齐原图 (线性插值)
    resampler_img = sitk.ResampleImageFilter()
    resampler_img.SetReferenceImage(template_img)
    resampler_img.SetInterpolator(sitk.sitkLinear)
    resampler_img.SetDefaultPixelValue(0)
    resampler_img.SetTransform(align_transform)
    pre_aligned = resampler_img.Execute(pre_img)
    
    # 对齐 Mask (最近邻插值，保证标签不失真)
    resampler_mask = sitk.ResampleImageFilter()
    resampler_mask.SetReferenceImage(template_img)
    resampler_mask.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler_mask.SetDefaultPixelValue(0)
    resampler_mask.SetTransform(align_transform)
    mask_aligned = resampler_mask.Execute(mask_img)

    print("Step 2: Extracting tumor and generating Soft Mask...")
    # 把多类别的标签二值化 (所有 >0 的地方都是肿瘤)
    mask_bin = sitk.BinaryThreshold(mask_aligned, lowerThreshold=1.0, upperThreshold=255.0, insideValue=1, outsideValue=0)
    
    # 膨胀 (保证边缘被充分包裹)
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelRadius([2, 2, 2]) # 膨胀2个像素
    mask_dilated = dilate_filter.Execute(mask_bin)
    
    # 高斯平滑 (产生羽化边缘)
    # 先转成 float
    mask_float = sitk.Cast(mask_dilated, sitk.sitkFloat32)
    gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian.SetSigma(1.5)
    soft_mask = gaussian.Execute(mask_float)
    # Clamp 到 [0, 1] 之间
    soft_mask = sitk.Clamp(soft_mask, sitk.sitkFloat32, 0.0, 1.0)
    
    print("Step 3: Executing image subtraction (creating the void)...")
    pre_float = sitk.Cast(pre_aligned, sitk.sitkFloat32)
    # 公式：Hole_Image = Pre_Image * (1.0 - Soft_Mask)
    inverted_mask = sitk.Subtract(1.0, soft_mask)
    hole_img = sitk.Multiply(pre_float, inverted_mask)
    
    print("Step 4: Applying Median Filter and 3D Sobel Edge Detection...")
    # 中值滤波 (Radius=1 足够去除椒盐噪声且保留主结构)
    median_filter = sitk.MedianImageFilter()
    median_filter.SetRadius([1, 1, 1])
    smoothed_hole = median_filter.Execute(hole_img)
    
    # 转换为 0-255 的标准范围，否则固定数值的 Canny 阈值会失效！
    rescaler = sitk.RescaleIntensityImageFilter()
    rescaler.SetOutputMinimum(0.0)
    rescaler.SetOutputMaximum(255.0)
    smoothed_hole_255 = rescaler.Execute(smoothed_hole)
    
    # 替换为 3D Sobel (Gradient Magnitude) 算子
    # 不需设定生硬的阈值，输出连续的解剖梯度边界
    gradient_filter = sitk.GradientMagnitudeImageFilter()
    edges = gradient_filter.Execute(smoothed_hole_255)

    # 修复并适配显示：将连续梯度图标准化到 0-255 并转为 8-bit 格式
    rescaler_edge = sitk.RescaleIntensityImageFilter()
    rescaler_edge.SetOutputMinimum(0.0)
    rescaler_edge.SetOutputMaximum(255.0)
    edges = rescaler_edge.Execute(edges)
    edges = sitk.Cast(edges, sitk.sitkUInt8)

    
    print("Step 5: Saving files...")
    sitk.WriteImage(soft_mask, str(OUTPUT_DIR / f"{target_id}_01_soft_mask.nii.gz"))
    sitk.WriteImage(hole_img, str(OUTPUT_DIR / f"{target_id}_02_hole_image.nii.gz"))
    sitk.WriteImage(smoothed_hole, str(OUTPUT_DIR / f"{target_id}_03_median_filtered.nii.gz"))
    sitk.WriteImage(edges, str(OUTPUT_DIR / f"{target_id}_04_canny_edges.nii.gz"))
    
    print(f"Pipeline finished! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
