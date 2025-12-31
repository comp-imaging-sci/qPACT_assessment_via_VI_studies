#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for the 3D deep learning model used in the accompanying qPACT study.

This script implements the training and evaluation pipeline used in the paper
on virtual imaging studies for quantitative photoacoustic computed tomography.

Provenance:
  - The overall training structure was originally inspired by upstream
    CSA/CS²-Net training code authored by Lei Mou.
  - The implementation has been extensively modified, refactored, and adapted
    for the present task by Refik Mert Cam (PhD candidate, ECE, UIUC).

Model overview:
  - Shared 3D backbone with two task-specific heads:
      * segmentation head (binary mask), optimized with weighted BCE + Dice loss
      * regression head (continuous target), optimized with weighted MSE

"""
import os
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import datetime
import numpy as np

from model.csnet_3d_leaky_2_decoders import CSNet3D
from dataloader.pact_dataset import Data



import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

@dataclass
class TrainConfig:
    data_root: Path
    epochs: int = 800
    lr: float = 2e-5
    snapshot_every: int = 2
    validate_every: int = 1
    ckpt_dir: Path = Path("./checkpoints")
    batch_size: int = 2
    num_workers: int = 4
    resume: Optional[Path] = None
    seed: int = 0


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train CSA/CS-Net 3D.")
    parser.add_argument("--data-root", type=Path, default=Path("/shared/anastasio-s2/Phantom/Breast_phantom_UBP/"),
                        help="Root directory containing the dataset subfolders / .mat files.")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--ckpt-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--snapshot-every", type=int, default=2, help="Save model checkpoint every N epochs.")
    parser.add_argument("--validate-every", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None, help="Optional path to a checkpoint to resume from.")
    parser.add_argument("--seed", type=int, default=0)

    ns = parser.parse_args()
    return TrainConfig(
        data_root=ns.data_root,
        epochs=ns.epochs,
        lr=ns.lr,
        snapshot_every=ns.snapshot_every,
        validate_every=ns.validate_every,
        ckpt_dir=ns.ckpt_dir,
        batch_size=ns.batch_size,
        num_workers=ns.num_workers,
        resume=ns.resume,
        seed=ns.seed,
    )
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1e-5  # Small constant to avoid division by zero
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        
        intersection = (input_flat * target_flat).sum()
        union = input_flat.sum() + target_flat.sum()
        
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice_score
        
        return dice_loss
        
def save_ckpt(net: nn.Module, step: int, ckpt_dir: Path) -> Path:
    """Save a model checkpoint and return the path."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d-")
    state_dict = net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()
    ckpt_path = ckpt_dir / f"CSNet3D_{date}{step}.pkl"
    torch.save(state_dict, ckpt_path)
    print(f"✔ Saved model to: {ckpt_path}")
    return ckpt_path
def load_ckpt(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    # If the model was saved using DataParallel, remove 'module.' from keys
    if 'module.' in list(checkpoint.keys())[0]:
        new_state_dict = {}
        for key, value in checkpoint.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)

    return model


# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_class_weights(mask):
    batch_size = mask.size(0)
    class_weights_batch = []
    
    mask_int = mask.long()
    for i in range(batch_size):
        mask_np = mask_int[i].cpu().numpy()
        class_counts = np.bincount(mask_np.flatten(), minlength=2)
        total_count = mask_np.size
        class_weights = total_count / (class_counts)
        #class_weights = class_weights / np.sum(class_weights)  # Normalize weights
        class_weights_batch.append(class_weights)
    
    class_weights_batch = np.array(class_weights_batch)
    #print(class_weights_batch)
    return torch.tensor(class_weights_batch, dtype=torch.float32).cuda() 


# Define the weighted binary cross entropy loss function
class WeightedBCELoss_new(nn.Module):
    def __init__(self):
        super(WeightedBCELoss_new, self).__init__()

    def forward(self, inputs, targets, weights):
        # Apply sigmoid to inputs
        #inputs = torch.sigmoid(inputs)
        inputs = torch.clamp(inputs,min=1e-7,max=1-1e-7)
        
        #weights = weights.view(weights.size(0), weights.size(1), 1, 1, 1)
        # Compute the binary cross entropy with weights

        bce_loss = - weights * (targets * torch.log(inputs + 1e-6) + 
                      ((1 - targets) * torch.log(1 - inputs + 1e-6)))

        return bce_loss.mean()        



def train(cfg: TrainConfig) -> None:
    # Start training
    print("\033[1;30;44m {} Start training ... {}\033[0m".format("*" * 8, "*" * 8))

    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    
    
    net = CSNet3D(classes=1, channels=3).cuda()
if cfg.resume is not None:
    print(f"Resuming from checkpoint: {cfg.resume}")
    load_ckpt(net, str(cfg.resume))
    
    
    #net = CSNet3D(classes=1, channels=3).cuda()
    net = nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).cuda()
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    
    # load train dataset
    train_data = Data(str(cfg.data_root), train=True, val=False, test=False)
    batchs_data = DataLoader(train_data, batch_size=cfg.batch_size, num_workers=4, shuffle=True)

    validation_data = Data(str(cfg.data_root), train=False, val=True, test=False)
    validation_batchs_data = DataLoader(validation_data, batch_size=cfg.batch_size, num_workers=4, shuffle=False)

    training_losses = []  # List to store training losses per epoch
    validation_losses = []  # List to store validation losses per epoch
    training_losses_reg = []
    training_losses_seg = []
    validation_losses_reg = []
    validation_losses_seg = []
    training_bce_losses = []
    training_dice_losses = []
    validation_bce_losses = []
    validation_dice_losses = []
    #training_losses_mse = []
    #validation_losses_mse = []

    # Define a learning rate scheduler
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.8)

    num_classes = 2

    for epoch in range(cfg.epochs):
        net.train()
        epoch_training_loss = 0.0
        epoch_training_loss_reg = 0.0
        epoch_training_loss_seg = 0.0
        epoch_training_bce_loss = 0.0
        epoch_training_dice_loss = 0.0
        #epoch_training_loss_mse = 0.0
        count_e = 0

        # Adjust the learning rate at every epoch
        # scheduler.step()

        for idx, batch in enumerate(batchs_data):
            
            if batch == 0:
                count_e += 1
                print("Error opening the file")
                continue
            else:
                image = batch[0].cuda()
                label = batch[1].cuda()
                weight = batch[2].cuda()
                mask = batch[3].cuda()
                weight_bce = batch[4].cuda()

                optimizer.zero_grad()
                pred_seg, pred_reg = net(image)
                pred_seg_shape = pred_seg.shape
                pred_reg_shape = pred_reg.shape


                loss_reg = torch.mean(weight * torch.square(pred_reg - label))

                #class_weights = compute_class_weights(mask)

                criterion_bce = WeightedBCELoss_new()
                criterion_dice = DiceLoss()
                #bce_loss = criterion_bce(pred_seg, mask, class_weights)
                bce_loss = criterion_bce(pred_seg, mask, weight_bce)
                dice_loss = criterion_dice(pred_seg, mask)

                loss_seg = 0.03 * bce_loss + 0.05 * dice_loss


                #loss_mse = torch.mean(torch.square(pred_reg - label))
                
                loss = loss_reg + loss_seg
                loss.backward()
                optimizer.step()

                #epoch_training_loss += loss.item()
                #epoch_training_loss_mse += loss_mse.item()

                epoch_training_loss += loss.item()
                epoch_training_loss_reg += loss_reg.item()
                epoch_training_loss_seg += loss_seg.item()
                epoch_training_bce_loss += bce_loss.item()
                epoch_training_dice_loss += dice_loss.item()

        training_losses.append(epoch_training_loss / len(batchs_data))
        training_losses_reg.append(epoch_training_loss_reg / len(batchs_data))
        training_losses_seg.append(epoch_training_loss_seg / len(batchs_data))
        training_bce_losses.append(epoch_training_bce_loss / len(batchs_data))
        training_dice_losses.append(epoch_training_dice_loss / len(batchs_data))
    
        print(f"Epoch [{epoch+1}/{cfg.epochs}], Training Loss: {training_losses[-1]}, "
              f"Reg Loss: {training_losses_reg[-1]}, Seg Loss: {training_losses_seg[-1]}, "
              f"BCE Loss: {training_bce_losses[-1]}, Dice Loss: {training_dice_losses[-1]}")


        #adjust_lr(optimizer, base_lr=cfg.lr, iter=epoch, max_iter=cfg.epochs, power=0.9)

        if (epoch + 1) % cfg.snapshot_every == 0:
            save_ckpt(net, epoch + 1, cfg.ckpt_dir)

        validation_loss, validation_loss_reg, validation_loss_seg, validation_bce_loss, validation_dice_loss = validate(net, validation_batchs_data)
        validation_losses.append(validation_loss)
        validation_losses_reg.append(validation_loss_reg)
        validation_losses_seg.append(validation_loss_seg)
        validation_bce_losses.append(validation_bce_loss)
        validation_dice_losses.append(validation_dice_loss)

        print(f"Epoch [{epoch+1}/{cfg.epochs}], Validation Loss: {validation_losses[-1]}, "
              f"Reg Loss: {validation_losses_reg[-1]}, Seg Loss: {validation_losses_seg[-1]}, "
              f"BCE Loss: {validation_bce_losses[-1]}, Dice Loss: {validation_dice_losses[-1]}")

        np.save(str(cfg.ckpt_dir) + '/' +'training_losses.npy', np.array(training_losses))
        np.save(str(cfg.ckpt_dir) + '/' +'validation_losses.npy', np.array(validation_losses))
        np.save(str(cfg.ckpt_dir) + '/' +'training_losses_reg.npy', np.array(training_losses_reg))
        np.save(str(cfg.ckpt_dir) + '/' +'validation_losses_reg.npy', np.array(validation_losses_reg))     
        np.save(str(cfg.ckpt_dir) + '/' +'training_losses_seg.npy', np.array(training_losses_seg))
        np.save(str(cfg.ckpt_dir) + '/' +'validation_losses_seg.npy', np.array(validation_losses_seg))
        np.save(str(cfg.ckpt_dir) + '/' +'training_bce_losses.npy', np.array(training_bce_losses))
        np.save(str(cfg.ckpt_dir) + '/' +'training_dice_losses.npy', np.array(training_dice_losses))
        np.save(str(cfg.ckpt_dir) + '/' +'validation_bce_losses.npy', np.array(validation_bce_losses))
        np.save(str(cfg.ckpt_dir) + '/' +'validation_dice_losses.npy', np.array(validation_dice_losses))










def validate(net, validation_batchs_data) -> tuple[float, float, float, float, float]:

    net.eval()
    validation_loss = 0.0
    validation_loss_mse = 0.0
    validation_loss = 0.0
    validation_loss_reg = 0.0
    validation_loss_seg = 0.0
    validation_bce_loss = 0.0
    validation_dice_loss = 0.0
    num_classes = 2

    with torch.no_grad():
        count_e = 0
        for idx, batch in enumerate(validation_batchs_data):
            if batch == 0:
                count_e += 1
                print("Error opening the file:")
                continue
            else:
                image = batch[0].cuda()
                label = batch[1].cuda()
                weight = batch[2].cuda()
                mask = batch[3].cuda()
                weight_bce = batch[4].cuda()

                pred_seg, pred_reg = net(image)
                
                loss_reg = torch.mean(weight * torch.square(pred_reg - label))
                
                # Compute class weights based on mask
                class_weights = compute_class_weights(mask)


                criterion_bce = WeightedBCELoss_new()
                criterion_dice = DiceLoss()
                #bce_loss = criterion_bce(pred_seg, mask, class_weights)
                bce_loss = criterion_bce(pred_seg, mask, weight_bce)
                dice_loss = criterion_dice(pred_seg, mask)

                loss_seg = 0.03 * bce_loss + 0.05 * dice_loss
                loss = loss_reg + loss_seg



                validation_loss += loss.item()
                validation_loss_reg += loss_reg.item()
                validation_loss_seg += loss_seg.item()
                validation_bce_loss += bce_loss.item()
                validation_dice_loss += dice_loss.item()

            """
                except OSError as e:        
                count_e += 1
                print("Error opening the file:", str(e))
                continue  # Move to the next file in the loop
            """

    return validation_loss / len(validation_batchs_data), \
            validation_loss_reg / len(validation_batchs_data), \
            validation_loss_seg / len(validation_batchs_data), \
            validation_bce_loss / len(validation_batchs_data), \
            validation_dice_loss / len(validation_batchs_data)



if __name__ == '__main__':
    cfg = parse_args()
    train(cfg)        
