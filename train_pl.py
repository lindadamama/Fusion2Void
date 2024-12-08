import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import seed_everything
import os
import argparse
import numpy as np
from Datasets.datasets import *
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torchvision
import json
import shutil
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import yaml
from easydict import EasyDict
from models.models import MODELS
from utils.optimizer import Lion
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
import collections
from utils.ema import EMA as EMACallback

# create folder
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# select dataset type
__dataset_train__ = {
    "MFI-WHU-train": Data_train
}
__dataset_val__ = {
    "MFI-WHU-test": Data_eval
}

class CoolSystem(pl.LightningModule):
    def __init__(self):
        """初始化训练的参数"""
        super(CoolSystem, self).__init__()
        # train datasets
        self.train_datasets = __dataset_train__[config["train_dataset"]](config)
        self.train_batchsize = config["train_batch_size"]
        # val datasets
        self.validation_datasets = __dataset_val__[config["test_dataset"]](config)
        self.val_batchsize = config["val_batch_size"]
        self.num_workers = config["num_workers"]
        self.save_path = PATH + config["save_path"]
        self.save_path_eval = PATH + config["save_path_eval"]

        ensure_dir(self.save_path)
        ensure_dir(self.save_path_eval)
        # set model type
        self.FusionNet = MODELS[config["fusion_net"]](config)
        self.RecNet = MODELS[config["rec_net"]](config)

        # loss
        self.criterionL1 = torch.nn.L1Loss()
        
        # Resume from pth ...
        if args.resume is not None:
            print("Loading from existing FusionNet checkpoint")
            ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
            new_state_dict = collections.OrderedDict()
            new_state_dict2 = collections.OrderedDict()
            for k in ckpt['state_dict']:
                if k[:10] != 'FusionNet.':
                    continue
                name = k[10:]
                new_state_dict[name] = ckpt['state_dict'][k]
            for k in ckpt['state_dict']:
                if k[:7] != 'RecNet.':
                    continue
                name = k[7:]
                new_state_dict2[name] = ckpt['state_dict'][k]
            self.FusionNet.load_state_dict(new_state_dict, strict=True)
            self.RecNet.load_state_dict(new_state_dict2, strict=True)

        print(PATH)
        # print model summary.txt
        import sys
        original_stdout = sys.stdout 
        with open(PATH + "/model_summary.txt", 'w+') as f:
            sys.stdout = f
            print(f'\n{self.FusionNet}\n')
            print(f'\n*******************************\n')
            print(f'\n{self.RecNet}\n')
            sys.stdout = original_stdout 
        self.automatic_optimization = False

    def train_dataloader(self):
        train_loader = data.DataLoader(
            self.train_datasets,
            batch_size=self.train_batchsize,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        return train_loader
    
    def val_dataloader(self):
        val_loader = data.DataLoader(
            self.validation_datasets,
            batch_size=self.val_batchsize,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
        return val_loader

    def configure_optimizers(self):
        """配置优化器和学习率的调整策略"""
        # Setting up optimizer.
        self.initlr = config["optimizer"]["args"]["lr"] # initial learning rate
        self.weight_decay = config["optimizer"]["args"]["weight_decay"] # optimizer weight decay
        self.momentum = config["optimizer"]["args"]["momentum"]
        if config["optimizer"]["type"] == "SGD":
            optimizer = optim.SGD(
                self.FusionNet.parameters(), 
                lr=self.initlr, 
                momentum=self.momentum, 
                weight_decay=self.weight_decay
            )
        elif config["optimizer"]["type"] == "ADAM":
            optimizer = optim.Adam(
                self.FusionNet.parameters(), 
                lr=self.initlr,
                weight_decay=self.weight_decay,
                betas=[0.9, 0.999]
            )
            optimizer2 = optim.Adam(
                self.RecNet.parameters(), 
                lr=self.initlr,
                weight_decay=self.weight_decay,
                betas=[0.9, 0.999]
            )
        elif config["optimizer"]["type"] == "ADAMW":
            optimizer = optim.AdamW(
                self.FusionNet.parameters(), 
                lr=self.initlr,
                weight_decay=self.weight_decay,
                betas=[0.9, 0.999]
            )   
        elif config["optimizer"]["type"] == "Lion":
            optimizer = Lion(
                filter(lambda p: p.requires_grad, self.FusionNet.parameters()), 
                lr=self.initlr,
                betas=[0.9, 0.99],
                weight_decay=0
            )
        else:
            exit("Undefined optimizer type")
        
        # Learning rate scheduler
        if config["optimizer"]["scheduler"] == "StepLR":
            step_size = config["optimizer"]["scheduler_set"]["step_size"]
            gamma = config["optimizer"]["scheduler_set"]["gamma"]
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif config["optimizer"]["scheduler"] == "CyclicLR":
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.initlr, max_lr=1.2*self.initlr, cycle_momentum=False)
        elif config["optimizer"]["scheduler"] == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["trainer"]["total_epochs"], eta_min=self.initlr * 1e-2)
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=config["trainer"]["total_epochs"], eta_min=self.initlr * 1e-2)
        else:
            scheduler = None
        return [optimizer, optimizer2], [scheduler, scheduler2]
    
    def training_step(self, data):
        """optimize the training"""
        opt1, opt2 = self.optimizers()
        opt1.zero_grad()
        opt2.zero_grad()
        device = next(self.FusionNet.parameters()).device
        
        """training step"""
        # Reading data.
        real_A, real_B, file = data
        H, W = real_A.shape[2:]
        # Fusion network
        qq = random.randint(0, 1)
        if qq == 0:
            fusion = self.FusionNet(real_A, real_B)
        else:
            fusion = self.FusionNet(real_B, real_A)
        
        block_size = config["mask_block"]
        ratio = config["mask_ratio"]
        mask = torch.ones([1, 1, int(H/block_size), int(W/block_size)]) * (1-ratio)
        mask = torch.bernoulli(mask).to(device)
        mask = F.interpolate(mask, size=(H, W), mode='nearest')

        A_mm = real_A * mask
        B_mm = real_B * mask
        
        # reconstruction network
        ss = random.randint(0, 1)
        if ss == 0:
            rec_A, rec_B = self.RecNet(A_mm, B_mm, fusion)
        else:
            rec_B, rec_A = self.RecNet(B_mm, A_mm, fusion)

        '''compute loss function'''
        z = torch.abs(fusion - real_B)
        loss_fidelity = self.criterionL1(fusion * z, real_A * z) 
        loss_recA = self.criterionL1(rec_A, real_A) 
        loss_recB = self.criterionL1(rec_B, real_B) 
        loss = 10 * loss_fidelity + loss_recA + loss_recB

        ######### Computing loss #########
        self.log('train_loss', loss, prog_bar=True)
        # self.log('lr', opt1.state_dict()['param_groups'][0]['lr'], sync_dist=True, prog_bar=True)
        
        '''clip gradients'''
        self.manual_backward(loss)
        opt1.step()
        opt2.step()
        
        # multiple schedulers
        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()

        return {'loss': loss}
    
    def on_validation_epoch_start(self):
        self.pred_dic = {}
        return super().on_validation_epoch_start()

    def validation_step(self, data, batch_idx):
        self.FusionNet.eval()
        device = next(self.FusionNet.parameters()).device

        """validation step"""
        real_A, real_B, file = data
        H, W = real_A.shape[2:]
        fusion = self.FusionNet(real_A, real_B)
        fusion1 = self.FusionNet(real_B, real_A)
        fusion = (fusion1 + fusion) / 2

        # Ensure the directory exists
        save_dir = os.path.join(self.save_path_eval, os.path.dirname(file[0]))
        os.makedirs(save_dir, exist_ok=True)

        save_image(fusion, os.path.join(self.save_path_eval, file[0]))

        block_size = config["mask_block"]
        ratio = config["mask_ratio"]
        mask = torch.ones([1, 1, int(H/block_size), int(W/block_size)]) * (1-ratio)
        mask = torch.bernoulli(mask).to(device)
        mask = F.interpolate(mask, size=(H, W), mode='nearest')

        A_mm = real_A * mask
        B_mm = real_B * mask
        
        # reconstruction network
        ss = random.randint(0, 1)
        if ss == 0:
            rec_A, rec_B = self.RecNet(A_mm, B_mm, fusion)
        else:
            rec_B, rec_A = self.RecNet(B_mm, A_mm, fusion)

        '''compute loss function'''
        z = torch.abs(fusion - real_B)
        loss_fidelity = self.criterionL1(fusion * z, real_A * z) 
        loss_recA = self.criterionL1(rec_A, real_A) 
        loss_recB = self.criterionL1(rec_B, real_B) 
        loss = 10 * loss_fidelity + loss_recA + loss_recB
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

        return {"val_loss": loss}

def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='./configs/Train_MMnet.json', type=str, help='Path to the config file')
    parser.add_argument('-r', '--resume', default=None, type=str, help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-r1', '--resume_ckpt', default=None, type=str, help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=[0,1], type=list, help='indices of GPUs to enable (default: all)')
    parser.add_argument('-v', '--val', default=False, type=bool, help='Validation')
    parser.add_argument('-val_path', default=None, type=str, help='Path to the val path')

    global args
    args = parser.parse_args()
    # set resume
    global config
    config = json.load(open(args.config))

    # Set seeds.
    seed = 42 # Global seed set to 42
    seed_everything(seed)
    
    # wandb log init
    global wandb_logger
    output_dir = './TensorBoardLogs'
    logger = TensorBoardLogger(name=config['name'] + "_" + config["train_dataset"] + "_" + config["test_dataset"], save_dir=output_dir)

    # Setting up path
    global PATH
    PATH = "./" + config["experim_name"] + "/" + config["train_dataset"] + "/" + config["test_dataset"] + "/" + str(config["tags"])
    ensure_dir(PATH + "/")
    shutil.copy2(args.config, PATH)
    
    # init pytorch-lightning
    ddp = DDPStrategy(process_group_backend="gloo", find_unused_parameters=True)
    model = CoolSystem()
    
    # set checkpoint mode and init ModelCheckpointHook
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=PATH,
        filename='best_model-epoch:{epoch:02d}',
        auto_insert_metric_name=False,   
        every_n_epochs=config["trainer"]["test_freq"],
        save_on_train_epoch_end=True,
        save_top_k=6,
        mode="min"
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    ema_callback = EMACallback(decay=0.996, every_n_steps=1)

    trainer = pl.Trainer(
        strategy=ddp,
        max_epochs=config["trainer"]["total_epochs"],
        accelerator='gpu', devices=args.device,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor_callback, ema_callback],
        check_val_every_n_epoch=config["trainer"]["test_freq"],
        log_every_n_steps=20,
    )   
    
    if args.val:
        trainer.validate(model, ckpt_path=args.val_path)
    else:
        trainer.fit(model)

if __name__ == '__main__':
    print('-----------------------------------------train_pl.py training-----------------------------------------')
    main()