import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# ★ 关键修改 1：把你的 BraTsdataset 文件夹动态加入系统路径，避开连字符无法 import 的报错
sys.path.append(os.path.join(project_root, "BraTsdataset"))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
# 注意：这里我们不再导入原作者的 VQGANDataset
from AutoEncoder.model.PatchVolume import patchvolumeAE
from train.callbacks import VolumeLogger
import argparse
from omegaconf import OmegaConf

# ★ 关键修改 2：导入你昨天写好的专属 DataLoader
from BraTsdataset.brats_dataset import get_brats_dataloader

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    pl.seed_everything(cfg.model.seed)

    print("🚀 正在加载自定义的 BraTS 数据集...")

    # 获取你在 yaml 配置文件里填写的 json 路径
    json_path = cfg.dataset.root_dir

    # ★ 关键修改 3：直接使用你的 DataLoader 替换原有逻辑
    # 注意：因为你的数据集目前只有 198 个，为了让 AutoEncoder 充分学习重构特征，
    # 我们可以暂时把 train 和 val 都指向这批数据。
    train_dataloader = get_brats_dataloader(json_path, batch_size=cfg.model.batch_size)
    val_dataloader = get_brats_dataloader(json_path, batch_size=1)

    bs, lr, ngpu = cfg.model.batch_size, cfg.model.lr, cfg.model.gpus
    print("Setting learning rate to {:.2e}, batch size to {}, ngpu to {}".format(lr, bs, ngpu))

    model = patchvolumeAE(cfg)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss',
                                     save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000,
                                     save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1,
                                     filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'))
    callbacks.append(VolumeLogger(
        batch_frequency=1500, max_volumes=4, clamp=True))

    logger = TensorBoardLogger(cfg.model.default_root_dir, name="my_model")
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=cfg.model.gpus,
        default_root_dir=cfg.model.default_root_dir,
        #strategy='ddp_find_unused_parameters_true',
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        check_val_every_n_epoch=2,
        num_sanity_val_steps=0,
        logger=logger
    )

    if cfg.model.resume_from_checkpoint and os.path.exists(cfg.model.resume_from_checkpoint):
        print('will start from the recent ckpt %s' % cfg.model.resume_from_checkpoint)
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=cfg.model.resume_from_checkpoint)
    else:
        trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)