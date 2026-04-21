import json
import os

# 1. 设置路径
json_path = "/mnt/proj/3D_MedDiffusion/BraTsdataset/brats_2024_pairs.json"
# 2. 你的图片实际存放位置（对应截图里的 BraTsdataset/data）
new_data_root = "/mnt/proj/3D_MedDiffusion/BraTsdataset/data"

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    for key in ["pre_op_img", "post_op_img", "post_op_seg"]:
        # 获取原始文件名，去掉 Windows 的路径前缀
        filename = os.path.basename(item[key].replace('\\', '/'))
        # 拼接为云端的绝对路径
        item[key] = os.path.join(new_data_root, filename)

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("✅ JSON 内部图片路径已更新为 Linux 绝对路径！")