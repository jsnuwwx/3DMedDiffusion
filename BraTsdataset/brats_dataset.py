import json
import os
import torch
from torch.utils.data import DataLoader
from monai.data import Dataset
from monai.transforms import CopyItemsd
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    SpatialPadd,
    Resized,
    ScaleIntensityd,
    Lambdad,
    EnsureTyped
)


# ★ 新增：将提取 Label 4 的逻辑写成一个普通的全局函数，Windows 就能识别了
def extract_label_4(x):
    return (x == 4).float()


def get_brats_dataloader(json_path, batch_size=1):
    with open(json_path, 'r', encoding='utf-8') as f:
        data_dicts = json.load(f)

    # 🚀 自动处理 Linux -> Windows 路径映射
    for item in data_dicts:
        for k in item:
            if isinstance(item[k], str) and "/mnt/proj/3D_MedDiffusion/" in item[k]:
                # 获取当前项目根目录
                root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                item[k] = item[k].replace("/mnt/proj/3D_MedDiffusion/", root + os.sep)

    transforms = Compose([
        LoadImaged(keys=["pre_op_img", "post_op_img", "post_op_seg"]),

        # 1. 复制分身
        CopyItemsd(keys=["post_op_img"], times=2, names=["image", "data"]),

        EnsureChannelFirstd(keys=["pre_op_img", "post_op_img", "post_op_seg", "image", "data"]),
        Orientationd(keys=["pre_op_img", "post_op_img", "post_op_seg", "image", "data"], axcodes="RAS"),

        # 2. ★ 核心修改：弃用 SpatialPadd，改用 Resized
        # 将所有维度统一缩放到 192，这是 PatchVolume 4x 最喜欢的立方体尺寸
        Resized(
            keys=["pre_op_img", "post_op_img", "post_op_seg", "image", "data"],
            spatial_size=(96, 96, 96),
            mode=("trilinear", "trilinear", "nearest", "trilinear", "trilinear")  # 掩码用最近邻插值
        ),

        ScaleIntensityd(keys=["pre_op_img", "post_op_img", "image", "data"], minv=-1.0, maxv=1.0),

        Lambdad(
            keys=["post_op_seg"],
            func=extract_label_4
        ),

        EnsureTyped(keys=["pre_op_img", "post_op_img", "post_op_seg", "image", "data"])
    ])

    dataset = Dataset(data=data_dicts, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    return dataloader


# --- 测试代码 ---
# --- 测试代码 ---
if __name__ == "__main__":
    import os

    # 自动获取当前 brats_dataset.py 所在的文件夹路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 自动拼接出 json 文件的绝对路径
    JSON_PATH = os.path.join(current_dir, "brats_2024_pairs.json")

    print(f"当前读取的 JSON 路径: {JSON_PATH}")
    print("正在构建 DataLoader...")
    loader = get_brats_dataloader(JSON_PATH, batch_size=1)

    print("尝试加载第一个 Batch 的数据...")
    for batch_data in loader:
        pre_img = batch_data["pre_op_img"]
        post_img = batch_data["post_op_img"]
        mask = batch_data["post_op_seg"]

        print("\n✅ 数据加载成功！查看张量形状:")
        print(f"术前图像形状: {pre_img.shape}")
        print(f"术后图像形状: {post_img.shape}")
        print(f"切除腔掩码形状: {mask.shape}")
        print(f"掩码的唯一值: {torch.unique(mask)}")
        break