
import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import conf


traindir = os.path.join(conf.dataset_dir, 'train')
valdir = os.path.join(conf.dataset_dir, 'val')
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

def train_loader():
  train_dataset = datasets.ImageFolder(
      traindir,
      transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
      ]))
  return torch.utils.data.DataLoader(train_dataset,
      batch_size=conf.train_batch_size, shuffle=True,
      num_workers=4, pin_memory=True)

def val_loader():
  val_dataset = datasets.ImageFolder(
      valdir,
      transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          normalize,
      ]))
  return torch.utils.data.DataLoader(val_dataset,
                                     batch_size=conf.val_batch_size,
                                     shuffle=True,
                                     num_workers=4)

if __name__ == '__main__':
  val_dataset = val_loader()
  for inputs, targets in val_dataset:
    print(inputs)
