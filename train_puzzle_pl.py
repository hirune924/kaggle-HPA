####################
# Import Libraries
####################
import os
import sys
import cv2
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning import loggers
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
import albumentations as A
import timm
from omegaconf import OmegaConf

####################
# Config
####################

conf_dict = {'batch_size': 32, 
             'epoch': 20, 
            'output_dir': '/kqi/output'}
conf_base = OmegaConf.create(conf_dict)


####################
# Dataset
####################

def load_image(img_dir, img_id):
    img_path = os.path.join(img_dir, img_id)
    img = []
    for c in ['blue', 'red', 'green', 'yellow']:
        img.append(cv2.imread(img_path + f'_{c}.png', cv2.IMREAD_GRAYSCALE))
    return np.stack(img, axis=-1)

def load_label(label, num_calsses):
    oh_label = torch.zeros(num_calsses)
    for i in label.split('|'):
        oh_label[int(i)] = 1
    return oh_label
    

class HPADataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        cv2.setNumThreads(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = load_image(self.data_dir, self.data.loc[idx, "ID"])
        if self.transform is not None:
            image = self.transform(image=image)
            image = torch.from_numpy(image["image"].transpose(2, 0, 1))
            
        label = load_label(self.data.loc[idx, "Label"], num_calsses=19)
        return image, label
           
####################
# Data Module
####################

class HPADataModule(pl.LightningDataModule):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        

    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage=None):
        if stage == 'fit':
            df = pd.read_csv('/kqi/parent/22019529/train.csv')

            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
            for fold, (train_index, val_index) in enumerate(kf.split(df.values, df["Label"])):
                df.loc[val_index, "fold"] = int(fold)
            df["fold"] = df["fold"].astype(int)

            train_df = df[df["fold"] != 0]
            valid_df = df[df["fold"] == 0]

            train_transform = A.Compose([
                        A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0), ratio=(1, 1), interpolation=1, always_apply=False, p=1.0),
                        A.Flip(always_apply=False, p=0.5),
                        #A.RandomGridShuffle(grid=(4, 4), always_apply=False, p=1.0),
                        #A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, always_apply=False, p=0.5),
                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
                        A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5),
                        #A.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
                        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0, rotate_limit=45, interpolation=1, border_mode=4, value=255, mask_value=None, always_apply=False, p=0.5),
                        A.Normalize(mean=(0.485, 0.456, 0.406, 0.406), std=(0.229, 0.224, 0.225, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                        A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
                        ])

            valid_transform = A.Compose([
                        A.Resize(height=512, width=512, interpolation=1, always_apply=False, p=1.0),
                        A.Normalize(mean=(0.485, 0.456, 0.406, 0.406), std=(0.229, 0.224, 0.225, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                        ])

            self.train_dataset = HPADataset(train_df, '/kqi/parent/22019529/train', transform=train_transform)
            self.valid_dataset = HPADataset(valid_df, '/kqi/parent/22019529/train', transform=valid_transform)
        elif stage == 'test':
            self.test_dataset = None
            pass
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=False)
        
####################
# Lightning Module
####################
from core.puzzle_utils import *
from core.networks import *
from tools.ai.torch_utils import *
from tools.ai.optim_utils import *

class LitSystem(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        #self.conf = conf
        self.save_hyperparameters(conf)
        self.model = Classifier(model_name='resnest50', num_classes=19, mode='normal')
        self.param_groups = self.model.get_parameter_groups(print_fn=None)
        self.gap_fn = self.model.global_average_pooling_2d
        
        self.class_loss_fn = nn.MultiLabelSoftMarginLoss(reduction='none')
        self.re_loss_fn = L1_Loss

    def forward(self, x):
        # use forward for inference/predictions
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optimizer = PolyOptimizer([
        {'params': self.param_groups[0], 'lr': 0.1, 'weight_decay': 1e-4},
        {'params': self.param_groups[1], 'lr': 2*0.1, 'weight_decay': 0},
        {'params': self.param_groups[2], 'lr': 10*0.1, 'weight_decay': 1e-4},
        {'params': self.param_groups[3], 'lr': 20*0.1, 'weight_decay': 0},
        ], lr=0.1, momentum=0.9, weight_decay=1e-4, max_step=100000, nesterov=True)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        logits, features = self.model(images, with_cam=True)

        tiled_images = tile_features(images, 4)

        tiled_logits, tiled_features = self.model(tiled_images, with_cam=True)
        
        re_features = merge_features(tiled_features, 4, self.hparams.batch_size)
        
        # Loss
        class_loss = self.class_loss_fn(logits, labels).mean()
        p_class_loss = self.class_loss_fn(self.gap_fn(re_features), labels).mean()
        
        class_mask = labels.unsqueeze(2).unsqueeze(3)
        re_loss = self.re_loss_fn(features, re_features) * class_mask
        re_loss = re_loss.mean()
        
        upper_alpha = 4.0
        alpha_schedule = 0.5
        alpha = min(upper_alpha * self.current_epoch+1 / (self.hparams.epoch * alpha_schedule), upper_alpha)
        loss = class_loss + p_class_loss + alpha * re_loss
        
        self.log('total_loss', loss, on_epoch=True)
        self.log('class_loss', class_loss, on_epoch=True)
        self.log('p_class_loss', p_class_loss, on_epoch=True)
        self.log('alpha', alpha, on_epoch=True)
        self.log('re_loss', re_loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.class_loss_fn(y_hat, y)
        
        return {
            "val_loss": loss,
            "y": y,
            "y_hat": y_hat
            }
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y = torch.cat([x["y"] for x in outputs]).cpu()
        y_hat = torch.cat([x["y_hat"] for x in outputs]).cpu()

        #preds = np.argmax(y_hat, axis=1)

        #val_accuracy = self.accuracy(y_hat, y)

        self.log('avg_val_loss', avg_val_loss)
        #self.log('val_acc', val_accuracy)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.class_loss_fn(y_hat, y)
        self.log('test_loss', loss)
        
####################
# Train
####################  
def main():
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf_base, conf_cli)
    print(OmegaConf.to_yaml(conf))
    seed_everything(2021)

    tb_logger = loggers.TensorBoardLogger(save_dir=os.path.join(conf.output_dir, 'tb_log/'))
    csv_logger = loggers.CSVLogger(save_dir=os.path.join(conf.output_dir, 'csv_log/'))

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(conf.output_dir, 'ckpt/'), monitor='avg_val_loss', 
                                          save_last=True, save_top_k=5, mode='min', 
                                          save_weights_only=True, filename='{epoch}-{avg_val_loss:.2f}')

    data_module = HPADataModule(conf)

    lit_model = LitSystem(conf)

    trainer = Trainer(
        logger=[tb_logger, csv_logger],
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=conf.epoch,
        gpus=-1,
        amp_backend='native',
        amp_level='O2',
        precision=16
            )

    trainer.fit(lit_model, data_module)

if __name__ == "__main__":
    main()