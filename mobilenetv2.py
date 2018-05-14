import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import conf


class Block(nn.Module):
  """
  expand + depthwise + pointwise
  """
  def __init__(self, in_planes, out_planes, expansion, stride, current_rate=1):
    super(Block, self).__init__()
    self.stride = stride


    if expansion > 1:
      planes = expansion * in_planes
      self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1,
          padding=0, bias=False)
      self.bn1 = nn.BatchNorm2d(planes)
    else:
      planes = in_planes
      self.conv1 = None
      self.bn1 = None
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        padding=current_rate, dilation=current_rate, groups=planes, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1,
        padding=0, bias=False)
    self.bn3 = nn.BatchNorm2d(out_planes)

    self.shortcut = False
    if stride == 1 and in_planes == out_planes:
      self.shortcut = True

  def forward(self, x):
    if self.conv1:
      out = F.relu6(self.bn1(self.conv1(x)))
    else:
      out = x
    out = F.relu6(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    if self.shortcut:
      out = out + x
    return out

class Mobilenetv2(nn.Module):
  # (expansion, out_planes, num_blocks, stride)
  cfg = [(1, 16, 1, 1),
         (6, 24, 2, 2),
         (6, 32, 3, 2),
         (6, 64, 4, 2),
         (6, 96, 3, 1),
         (6, 160, 3, 2),
         (6, 320, 1, 1)]

  def __init__(self, num_classes=conf.num_classes):
    super(Mobilenetv2, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(32)
    self.layers = self._make_layers(in_planes=32)
    self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn2 = nn.BatchNorm2d(1280)
    self.connect = nn.Conv2d(1280, num_classes, kernel_size=1, bias=True)

  def _make_layers(self, in_planes):
    layers = []
    for expansion, out_planes, num_blocks, stride in self.cfg:
      strides = [stride] + [1]*(num_blocks-1)
      for stride in strides:
        layers.append(Block(in_planes, out_planes, expansion, stride))
        in_planes = out_planes
    return nn.Sequential(*layers)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.layers(out)
    out = F.relu(self.bn2(self.conv2(out)))

    out = F.avg_pool2d(out, 7)
    # out = out.view(out.size(0), -1)
    out = self.connect(out)
    out = out.squeeze(3).squeeze(2)
    # return F.log_softmax(out, dim=1)
    return out

class Mobilenetv2_base(nn.Module):

  cfg = [(1, 16, 1, 1),
         (6, 24, 2, 2),
         (6, 32, 3, 2),
         (6, 64, 4, 2),
         (6, 96, 3, 1),
         (6, 160, 3, 2),
         (6, 320, 1, 1)]
  target_stride = 16

  def __init__(self):
    super(Mobilenetv2_base, self).__init__()
    # self.cfg = cfg

    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(32)
    self.layers = self._make_layers(in_planes=32)

  def _make_layers(self, in_planes):
    layers = []

    output_stride = 2
    current_rate = 1

    for expansion, out_planes, num_blocks, stride in self.cfg:
      if output_stride == self.target_stride:
        current_rate *= stride
        top_stride = 1
      else:
        top_stride = stride
        output_stride *= stride
      strides = [top_stride] + [1]*(num_blocks-1)
      for layer_stride in strides:
        layers.append(Block(in_planes, out_planes,
                            expansion, layer_stride, current_rate))
        in_planes = out_planes
    return nn.Sequential(*layers)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.layers(out)

    return out

class Deeplab(nn.Module):

  def __init__(self):
    super(Deeplab, self).__init__()
    self.base_net = Mobilenetv2_base()
    self.avgpool_connect = nn.Conv2d(320, 256, kernel_size=1, bias=False)
    self.avgpool_batchnorm = nn.BatchNorm2d(256)

    self.head_conv = nn.Conv2d(320, 256, kernel_size=1, bias=False)
    self.head_batchnorm = nn.BatchNorm2d(256)

    self.top_conv = nn.Conv2d(512, 256, kernel_size=1, bias=False)
    self.top_batchnorm = nn.BatchNorm2d(256)

    self.semantic_conv = nn.Conv2d(256, 21, kernel_size=1, bias=True)


  def forward(self, x):
    feature = self.base_net(x)

    avgpool = F.avg_pool2d(feature, 33)
    avgpool = self.avgpool_connect(avgpool)
    avgpool = F.relu(self.avgpool_batchnorm(avgpool))
    avgpool = F.upsample(avgpool, size=33, mode='bilinear')

    head = self.head_conv(feature)
    head = F.relu(self.head_batchnorm(head))

    top = torch.cat((avgpool, head), dim=1)
    top = self.top_conv(top)
    top = F.relu(self.top_batchnorm(top))

    semantic = self.semantic_conv(top)
    semantic = F.upsample(semantic, size=513, mode='bilinear')

    out = torch.argmax(semantic, dim=1)

    return out







def test():
  net = Mobilenetv2()
  x = Variable(torch.randn(2, 3, 224, 224))
  y = net(x)
  print(y.size())

def test_base():
  net = Mobilenetv2_base()
  x = Variable(torch.randn(2, 3, 224, 224))
  y = net(x)
  print(y.size())

def test_deeplab():
  net = Deeplab()
  x = Variable(torch.randn(2, 3, 513, 513))
  y = net(x)
  print(y.size())

if __name__ == '__main__':
  # test()
  # test_base()
  test_deeplab()
