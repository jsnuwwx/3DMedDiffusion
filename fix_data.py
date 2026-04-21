import json
import os

# 1. 你云端 JSON 的路径
json_path = "/mnt/proj/3D_MedDiffusion/BraTsdataset/dataset_cloud.json"
# 2. 你云端存放 .nii.gz 图片的文件夹路径
cloud_data_root = "/mnt/proj/3D_MedDiffusion/BraTsdataset/data"

with open(json_path, 'r') as f:
    data = json.load(f)

for item in data:
    for key in ["pre_op_img", "post_op_img", "post_op_seg"]:
        filename = os.path.basename(item[key])
        # 强制转换为云端绝对路径
        item[key] = os.path.join(cloud_data_root, filename)

with open(json_path, 'w') as f:
    json.dump(data, f, indent=4)

print("✅ JSON 路径已全部修正为云端绝对路径！")