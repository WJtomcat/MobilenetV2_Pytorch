
from dataset import train_loader, val_loader
from mobilenetv2 import Mobilenetv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils import progress_bar
import conf

from logger import Logger

logger = Logger('./logs')

def to_np(x):
  return x.data.cpu().numpy()


net = Mobilenetv2()
state_dict = torch.load('mobilenetv2_20.pth')

for k in list(state_dict.keys()):
  if 'module' in k:
    state_dict[k.replace('module.', '')] = state_dict[k]
    del state_dict[k]


net.load_state_dict(state_dict)
net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=conf.lr, momentum=0.9, weight_decay=5e-4)

net = torch.nn.DataParallel(net)
cudnn.benchmark = True


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)

# weights_normal_init(net)

def train(epoch, net, train_dataset):
  net.train()
  print('\nEpoch: %d' % epoch)
  train_loss = 0
  correct = 0
  total = 0
  for batch_idx, (inputs, targets) in enumerate(train_dataset):
    inputs, targets = inputs.cuda(), targets.cuda()
    targets = targets + 1
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
    accuracy = (targets == predicted).float().mean()

    info = {
        'loss': loss.data[0],
        'accuracy': accuracy
    }

    for tag, value in info.items():
      logger.scalar_summary(tag, value, batch_idx+1)

    # for tag, value in net.named_parameters():
    #   tag = tag.replace('.', '/')
    #   logger.histo_summary(tag, to_np(value), batch_idx+1)
    #   logger.histo_summary(tag+'/grad', to_np(value.grad), batch_idx+1)

    progress_bar(batch_idx, len(train_dataset), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    if batch_idx % 1000:
      n = batch_idx / 1000
      state = net.state_dict()
      torch.save(state, './mobilenetv2_2_' + str(n) + '.pth')



def test(epoch):
  net.eval()
  val_dataset = val_loader()
  test_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(val_dataset):
      inputs, targets = inputs.cuda(), targets.cuda()
      outputs = net(inputs)
      # print(targets)
      targets = targets + 1
      loss = criterion(outputs, targets)

      test_loss += loss.item()
      # _, predicted = outputs.max(1)
      # predicted = predicted - 1
      predicted = torch.argmax(outputs, dim=1)
      # predicted = predicted - 1
      total += targets.size(0)
      # print(predicted)
      correct += predicted.eq(targets).sum().item()

      print('batch_num: %d, test_loss: %.3f, Acc: %.3f%%'
            % (batch_idx, test_loss/(batch_idx+1), 100.*correct/total))




if __name__ == '__main__':
  # test(0)
  train_dataset = train_loader()
  train(0, net, train_dataset)
  # train(1)
