import torch
from torch.utils.data.dataset import Dataset
import os
import random
import glob
import torchio as tio
import json


class VQGANDataset_4x(Dataset):
    def __init__(self, root_dir=None, augmentation=False, split='train', stage=1, patch_size=64):
        randnum = 216
        self.file_names = []
        self.stage = stage
        self.patch_size = patch_size
        print(f"Loading dataset from: {root_dir}")

        if root_dir.endswith('json'):
            with open(root_dir) as json_file:
                dataroots = json.load(json_file)

            # 🚨 完美匹配你截图里的 JSON 格式 (List of Dicts)
            if isinstance(dataroots, list):
                for item in dataroots:
                    if "pre_op_img" in item:
                        raw_path = item["pre_op_img"]
                        # 确保路径存在才加入，防止报 num_samples=0 的错误
                        if os.path.exists(raw_path):
                            self.file_names.append(raw_path)
                        else:
                            print(f"⚠️ 找不到文件: {raw_path}")
            else:
                # 兼容原作者的旧版字典格式
                for key, value in dataroots.items():
                    if type(value) == list:
                        for path in value:
                            self.file_names.extend(glob.glob(os.path.join(path, './*.nii*'), recursive=True))
                    else:
                        self.file_names.extend(glob.glob(os.path.join(value, './*.nii*'), recursive=True))
        else:
            self.root_dir = root_dir
            self.file_names = glob.glob(os.path.join(root_dir, './*.nii*'), recursive=True)

        random.seed(randnum)
        random.shuffle(self.file_names)

        self.split = split
        self.augmentation = augmentation

        # 划分训练集和验证集
        total_files = len(self.file_names)
        if split == 'train':
            self.file_names = self.file_names[:-40] if total_files > 40 else self.file_names
        elif split == 'val':
            self.file_names = self.file_names[-40:] if total_files > 40 else []

        self.patch_sampler = tio.data.UniformSampler(patch_size)
        self.patch_sampler_256 = tio.data.UniformSampler((256, 256, 128))
        self.randomflip = tio.RandomFlip(axes=(0, 1), flip_probability=0.5)

        print(f'Dataset Split: {split} | Files Ready: {len(self.file_names)} | Stage: {stage} | Patch: {patch_size}')

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        path = self.file_names[index]
        whole_img = tio.ScalarImage(path)

        # 🚨 终极修复：不论是 train 还是 val，统一使用 patch_sampler 切出标准块！
        # 这样喂给模型的数据永远是完美的 patch_size（如 64x64x64 或 96x96x96）
        # 彻底杜绝 unfold 函数丢弃边缘导致的 size mismatch 报错！
        if self.split in ['train', 'val']:
            img = None
            while img is None or img.data.sum() == 0:
                img = next(self.patch_sampler(tio.Subject(image=whole_img)))['image']
        else:
            img = whole_img

        if self.augmentation:
            img = self.randomflip(img)

        imageout = img.data
        if self.augmentation and random.random() > 0.5:
            imageout = torch.rot90(imageout, dims=(1, 2))

        # 归一化到 [-1, 1]
        imageout = imageout * 2 - 1
        imageout = imageout.transpose(1, 3).transpose(2, 3)
        imageout = imageout.type(torch.float32)

        if self.split == 'val':
            return {'data': imageout, 'affine': img.affine, 'path': path}
        else:
            return {'data': imageout}