
from dataset import train_loader, val_loader
from mobilenetv2 import Mobilenetv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils import progress_bar
import conf

from logger import Logger

def to_np(x):
    return x.data.cpu().numpy()


def weights_normal_init(model):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)


def train(epoch, net, train_dataset, lr, logger):
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    net.train()
    print('\nEpoch: %d' % epoch)
    batch_len = len(train_dataset)
    prebatch_nums = epoch * batch_len
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataset):
        inputs, targets = inputs.cuda(), targets.cuda()
        targets = targets
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
            logger.scalar_summary(tag, value, prebatch_nums+batch_idx+1)

        progress_bar(batch_idx, len(train_dataset), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | epoch: %d'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, epoch))

    state = net.state_dict()
    torch.save(state, './mobilenetv2_' + str(epoch) + '.pth')



def test(epoch, net, val_dataset, logger):
    net.eval()
    print('\nEpoch: %d validation begin' % epoch)
    batch_len = len(val_dataset)
    prebatch_nums = epoch * batch_len
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_dataset):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            targets = targets
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = (targets == predicted).float().mean()

            info = {
                'valloss': loss.data[0],
                'valaccuracy': accuracy
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, prebatch_nums+batch_idx+1)

            progress_bar(batch_idx, batch_len, 'Loss: %.3f | Acc: %.3f%% (%d/%d) | epoch: %d'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, epoch))




if __name__ == '__main__':

    net = Mobilenetv2()

    # state_dict = torch.load('mobilenetv2_9.pth')
    # for k in list(state_dict.keys()):
    #     if 'module' in k:
    #         state_dict[k.replace('module.', '')] = state_dict[k]
    #         del state_dict[k]
    #
    # net.load_state_dict(state_dict)

    weights_normal_init(net)
    net = net.cuda()

    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

    train_dataset = train_loader()
    train_logger = Logger('./logs')

    val_dataset = val_loader()

    lr = 0.05

    for epoch in range(10):
        train(epoch, net, train_dataset, lr, train_logger)
        lr *= 0.8
        test(epoch, net, val_dataset, train_logger)

    # net = Mobilenetv2()
    #
    # state_dict = torch.load('mobilenetv2_9.pth')
    #
    # for k in list(state_dict.keys()):
    #     if 'module' in k:
    #         state_dict[k.replace('module.', '')] = state_dict[k]
    #         del state_dict[k]
    #
    # net.load_state_dict(state_dict)
    # net.cuda()
    #
    # val_dataset = val_loader()
    # test(net, val_dataset)
