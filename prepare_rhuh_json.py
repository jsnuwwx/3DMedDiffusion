import os
import json
import glob
import random
from pathlib import Path

def main():
    standardized_dir = Path(r"E:\deep learning code\3D_MedDiffusion_Code\RHUH_standardized")
    
    # 我们创建三个 JSON 文件：全部、训练集、测试集
    all_json = Path(r"E:\deep learning code\3D_MedDiffusion_Code\rhuh_all_pairs.json")
    train_json = Path(r"E:\deep learning code\3D_MedDiffusion_Code\rhuh_train_pairs.json")
    test_json = Path(r"E:\deep learning code\3D_MedDiffusion_Code\rhuh_test_pairs.json")

    # Pattern: RHUH-XXXX-100-t1c.nii.gz (Pre) and RHUH-XXXX-101-t1c.nii.gz (Post)
    pre_files = sorted(list(standardized_dir.glob("*-100-t1c.nii.gz")))
    
    pairs = []
    for pre_p in pre_files:
        p_id = pre_p.name.split("-100-")[0]
        post_p = standardized_dir / f"{p_id}-101-t1c.nii.gz"
        
        if post_p.exists():
            pairs.append({
                "p_id": p_id,
                "pre_op_img": str(pre_p),
                "post_op_img": str(post_p)
            })

    # 固定随机种子确保可重复性
    random.seed(42)
    random.shuffle(pairs)

    # 划分比例：35个训练，5个测试
    num_test = 5
    test_pairs = pairs[:num_test]
    train_pairs = pairs[num_test:]

    # 保存文件
    with open(all_json, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=4)
    with open(train_json, 'w', encoding='utf-8') as f:
        json.dump(train_pairs, f, indent=4)
    with open(test_json, 'w', encoding='utf-8') as f:
        json.dump(test_pairs, f, indent=4)
        
    print(f"Dataset split complete:")
    print(f"   - Total cases: {len(pairs)}")
    print(f"   - Train set: {len(train_pairs)} (saved to {train_json.name})")
    print(f"   - Test set: {len(test_pairs)} (saved to {test_json.name})")
    print(f"   - Test Case IDs: {[p['p_id'] for p in test_pairs]}")

if __name__ == "__main__":
    main()
