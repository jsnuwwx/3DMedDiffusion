import os
import json
from pathlib import Path
from tqdm import tqdm


def find_file(folder_path, keyword, suffix_list=['.nii.gz', '.nii']):
    """在文件夹中寻找包含特定关键字且后缀匹配的文件"""
    for file in folder_path.glob("*"):
        if keyword in file.name.lower():
            if any(file.name.endswith(s) for s in suffix_list):
                return str(file.absolute())
    return None


def audit_brats_dataset(data_root):
    root = Path(data_root)
    all_folders = [f.name for f in root.iterdir() if f.is_dir()]

    patient_ids = set()
    for folder in all_folders:
        if "BraTS-GLI-" in folder:
            p_id = "-".join(folder.split("-")[:-1])
            patient_ids.add(p_id)

    dataset_pairs = []
    missing_report = []

    print(f"开始体检，共发现患者: {len(patient_ids)} 名")

    for p_id in tqdm(sorted(list(patient_ids))):
        pre_folder = root / f"{p_id}-100"
        post_folder = root / f"{p_id}-101"

        # 1. 检查文件夹是否存在
        if not pre_folder.exists():
            missing_report.append(f"{p_id} 失败: 缺少术前文件夹 (-100)")
            continue
        if not post_folder.exists():
            missing_report.append(f"{p_id} 失败: 缺少术后文件夹 (-101)")
            continue

        # 2. 查找关键文件
        pre_t1c = find_file(pre_folder, "t1c")
        post_t1c = find_file(post_folder, "t1c")
        post_seg = find_file(post_folder, "seg")

        # 3. 记录缺失的文件
        missing_files = []
        if not pre_t1c: missing_files.append("术前 t1c 图")
        if not post_t1c: missing_files.append("术后 t1c 图")
        if not post_seg: missing_files.append("术后 seg 掩码")

        if missing_files:
            missing_report.append(f"{p_id} 失败: 文件夹都有，但缺少 {', '.join(missing_files)}")
        else:
            # 文件齐全，配对成功！
            dataset_pairs.append({
                "id": p_id,
                "pre_op_img": pre_t1c,
                "post_op_img": post_t1c,
                "post_op_seg": post_seg
            })

    # 输出报告
    print("\n" + "=" * 40)
    print(f"✅ 成功配对并完美包含 100/101/seg 的患者: {len(dataset_pairs)}")
    print(f"❌ 数据残缺被排除的患者: {len(missing_report)}")
    print("=" * 40)

    if missing_report:
        print("\n--- 失败原因明细 ---")
        for line in missing_report[:20]:  # 只打印前20个避免刷屏
            print(line)
        if len(missing_report) > 20:
            print(f"... 以及其他 {len(missing_report) - 20} 名患者。")

        # 把完整的失败名单保存下来供你查看
        with open("../missing_report.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(missing_report))
        print("\n📄 完整的失败名单已保存至当前目录的 missing_report.txt")

    # 依然保存成功的那部分数据
    with open("brats_2024_pairs.json", 'w', encoding='utf-8') as f:
        json.dump(dataset_pairs, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    DATA_PATH = r"BraTsdataset/data"
    audit_brats_dataset(DATA_PATH)