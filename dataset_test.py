
from dataset import train_loader, val_loader

val_dataset = val_loader()
for i, (input, target) in enumerate(val_dataset):
  print(input.size())
  print(target.size())
