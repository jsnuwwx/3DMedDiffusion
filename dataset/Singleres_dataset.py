import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import json
import torchio as tio


from monai.transforms import Compose, LoadImaged, ScaleIntensityd, EnsureChannelFirstd, Resized

class Singleres_dataset(Dataset):
    def __init__(self, root_dir=None, resolution=[32, 32, 32], generate_latents=False):
        self.all_files = []
        self.resolution = resolution
        self.generate_latents = generate_latents
        
        # 定义统一的归一化 Transform
        keys = ["img"] if generate_latents else ["pre", "post"]
        self.transforms = Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Resized(keys=keys, spatial_size=tuple(self.resolution), mode="trilinear"),
            ScaleIntensityd(keys=keys, minv=-1.0, maxv=1.0)
        ])
        
        # 注意：Stage 3 训练时加载的是 .nii 的 Latents，
        # 原作者的代码里 latent 已经是 [-1, 1] 了（在生成时做的），
        # 所以只有 generate_latents=True 时才需要对原图做 ScaleIntensity。

        if root_dir.endswith('json'):
            with open(root_dir) as json_file:
                dataroots = json.load(json_file)

            if isinstance(dataroots, list):
                for item in dataroots:
                    # 🚀 自动处理 Linux -> Windows 路径映射
                    for k in item:
                        if isinstance(item[k], str) and "/mnt/proj/3D_MedDiffusion/" in item[k]:
                            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            item[k] = item[k].replace("/mnt/proj/3D_MedDiffusion/", root + os.sep)

                    if self.generate_latents:
                        # ==================================================
                        # 🚀 模式 A：提取特征模式 (提取术前和术后两张图)
                        # ==================================================
                        for key in ["pre_op_img", "post_op_img"]:
                            if key in item:
                                file_path = item[key]
                                if os.path.exists(file_path):
                                    self.all_files.append({"img": file_path})
                    else:
                        # ==================================================
                        # 🧠 模式 B：Stage 3 训练模式 (任务：术前生术后)
                        # ==================================================
                        if "pre_op_img" in item and "post_op_img" in item:
                            self.all_files.append({
                                "pre": item["pre_op_img"],
                                "post": item["post_op_img"]
                            })
            else:
                # 兼容原作者旧字典格式
                for key, value in dataroots.items():
                    if not generate_latents:
                        value = value + '_latents'
                    file_paths = glob.glob(value + '/*.nii*', recursive=True)
                    for fp in file_paths:
                        self.all_files.append({key: fp})

        self.file_num = len(self.all_files)
        print(f"Total files found: {self.file_num}")

    def __len__(self):
        return getattr(self, 'file_num', 0)

    def __getitem__(self, index):
        if self.generate_latents:
            # --------------------------------------------------
            # 🚀 模式 A 动作：读取原图，扔给 AE 压缩
            # --------------------------------------------------
            file_path = list(self.all_files[index].values())[0]
            # 使用 monai 加载以确保一致性 (tio 也可以，但 monai transforms 更方便)
            data_dict = self.transforms({"img": file_path})
            return data_dict["img"], file_path
        else:
            # --------------------------------------------------
            # 🧠 模式 B 动作：成对读取 Latents，扔给 Diffusion
            # --------------------------------------------------
            data_dict = self.all_files[index]

            # 1. 组装术后特征 (Target) 的路径并以 tio 格式读取
            post_path = data_dict["post"]
            dir_name_post = os.path.basename(os.path.dirname(post_path))
            post_latent_path = post_path.replace(dir_name_post, dir_name_post + '_latents')
            post_latent_io = tio.ScalarImage(post_latent_path)

            # 2. 组装术前特征 (Hint/Condition) 的路径并以 tio 格式读取
            pre_path = data_dict["pre"]
            dir_name_pre = os.path.basename(os.path.dirname(pre_path))
            pre_latent_path = pre_path.replace(dir_name_pre, dir_name_pre + '_latents')
            pre_latent_io = tio.ScalarImage(pre_latent_path)

            # 🛠️ 核心修正：无论磁盘上的 Latent 是多大（比如 24x24），都强制 Resize 到请求的分辨率（比如 24x48）
            # 这能彻底解决采样阶段的 concat 维度冲突
            transform = tio.Resize(self.resolution)
            post_latent = transform(post_latent_io).data.to(torch.float32)
            pre_latent = transform(pre_latent_io).data.to(torch.float32)

            # 注意：这里的 latent 已经是 [-1, 1] 了，不需要再做 ScaleIntensity
            return post_latent, pre_latent, torch.tensor(self.resolution, dtype=torch.float32) / 64.0