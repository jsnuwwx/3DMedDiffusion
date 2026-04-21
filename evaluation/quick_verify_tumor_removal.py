import os
import argparse
import numpy as np
import torch
import torchio as tio
import imageio
import json


def load_paths_from_json(json_path, index):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    item = data[index] if isinstance(data, list) else None
    pre = item.get("pre_op_img") if item else None
    post_seg = item.get("post_op_seg") if item else None
    return pre, post_seg


def to_numpy_xy(slice2d):
    arr = slice2d.astype(np.float32)
    arr = arr - arr.min()
    m = arr.max()
    if m > 0:
        arr = arr / m
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def center_slice_with_mask(img_tensor, mask_tensor):
    c, x, y, z = img_tensor.shape
    mask = mask_tensor[0] > 0.5
    coords = torch.nonzero(mask, as_tuple=False)
    if coords.numel() == 0:
        return img_tensor[0, :, :, z // 2].cpu().numpy(), mask_tensor[0, :, :, z // 2].cpu().numpy()
    cz = int(coords[:, 2].float().mean().item())
    cz = int(np.clip(cz, 0, z - 1))
    return img_tensor[0, :, :, cz].cpu().numpy(), mask_tensor[0, :, :, cz].cpu().numpy()


def overlay_mask(gray, mask):
    h, w = gray.shape
    rgb = np.stack([gray, gray, gray], axis=-1)
    edge = np.logical_xor(mask, np.logical_and(
        np.pad(mask[1:, :], ((0, 1), (0, 0)), constant_values=False),
        np.pad(mask[:-1, :], ((1, 0), (0, 0)), constant_values=False)))
    edge |= np.logical_xor(mask, np.logical_and(
        np.pad(mask[:, 1:], ((0, 0), (0, 1)), constant_values=False),
        np.pad(mask[:, :-1], ((0, 0), (1, 0)), constant_values=False)))
    rgb[edge, 0] = 255
    rgb[edge, 1] = 0
    rgb[edge, 2] = 0
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-json", type=str, required=True)
    parser.add_argument("--gen", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    pre_path, post_seg_path = load_paths_from_json(args.data_json, args.index)
    gen_img = tio.ScalarImage(args.gen)
    gen_t = gen_img.data.float()
    if pre_path is None or post_seg_path is None:
        raise RuntimeError("Missing pre_op_img or post_op_seg in json.")
    pre_img = tio.ScalarImage(pre_path)
    pre_resampled = tio.Resize(gen_t.shape[-3:])(pre_img)
    pre_t = pre_resampled.data.float()
    seg_img = tio.LabelMap(post_seg_path)
    seg_resampled = tio.Resize(gen_t.shape[-3:])(seg_img)
    seg_t = (seg_resampled.data > 0).float()

    m = seg_t > 0.5
    if m.sum() == 0:
        raise RuntimeError("Empty post_op_seg after resizing.")
    pre_in = pre_t[m.expand_as(pre_t)].mean().item()
    gen_in = gen_t[m.expand_as(gen_t)].mean().item()
    diff_in = pre_in - gen_in

    sl_pre, sl_mask = center_slice_with_mask(pre_t, seg_t)
    sl_gen, _ = center_slice_with_mask(gen_t, seg_t)
    sl_pre_u8 = to_numpy_xy(sl_pre)
    sl_gen_u8 = to_numpy_xy(sl_gen)
    ov_pre = overlay_mask(sl_pre_u8, sl_mask > 0.5)
    ov_gen = overlay_mask(sl_gen_u8, sl_mask > 0.5)
    pad = 10
    h = max(ov_pre.shape[0], ov_gen.shape[0])
    w = ov_pre.shape[1] + ov_gen.shape[1] + pad
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:ov_pre.shape[0], :ov_pre.shape[1]] = ov_pre
    canvas[:ov_gen.shape[0], ov_pre.shape[1] + pad:ov_pre.shape[1] + pad + ov_gen.shape[1]] = ov_gen
    out_path = args.out or os.path.join(os.path.dirname(args.gen), "quick_check.png")
    imageio.imwrite(out_path, canvas)
    print(f"pre_mask_mean={pre_in:.4f}, gen_mask_mean={gen_in:.4f}, delta(pre-gen)={diff_in:.4f}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

