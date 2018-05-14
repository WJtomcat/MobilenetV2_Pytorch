import tensorflow as tf
from collections import OrderedDict

from mobilenetv2 import Mobilenetv2
import torch

model_name = './tf_models/mobilenet_v2_1.0_224.ckpt'

reader = tf.train.NewCheckpointReader(model_name)

var_to_shape_map = reader.get_variable_to_shape_map()
var_dict = {k:reader.get_tensor(k) for k in var_to_shape_map.keys()}


for k in list(var_dict.keys()):
  if 'RMSProp' in k or 'Momentum' in k or 'ExponentialMovingAverage' in k or 'global_step' in k:
    del var_dict[k]



print(len(var_dict))

# for k in list(var_dict.keys()):
#   print(k)

# i = 0
# for k in list(var_dict.keys()):
#   # if k.find('expanded_conv'):
#   if 'expanded_conv' in k:
#     print k
#     i += 1
#
# print(i)

for k in list(var_dict.keys()):
  if 'expanded_conv' in k:
    var_dict['layers.'+k[k.find('/')+1:]] = var_dict[k]
    del var_dict[k]

dummy_replace = OrderedDict([
    ('expanded_conv_', ''),
    ('expanded_conv', '0'),
    ('/depthwise/depthwise_weights', '.conv2.weight'),
    ('/depthwise/BatchNorm', '.bn2'),
    ('/project/weights', '.conv3.weight'),
    ('/project/BatchNorm', '.bn3'),
    ('/expand/BatchNorm', '.bn1'),
    ('/expand/weights', '.conv1.weight'),
    ('/moving_mean', '.running_mean'),
    ('/moving_variance', '.running_var'),
    ('/gamma', '.weight'),
    ('/beta', '.bias'),
    ('MobilenetV2', ''),
    ('/Conv_1/BatchNorm', 'bn2'),
    ('/Conv_1', 'conv2'),
    ('/Conv/BatchNorm', 'bn1'),
    ('/Conv', 'conv1'),
    ('/weights', '.weight'),
    ('/Logitsconv12d_1c_1x1/biases', 'connect.bias'),
    ('/Logitsconv12d_1c_1x1.weight', 'connect.weight')
])
for a, b in dummy_replace.items():
    for k in list(var_dict.keys()):
        if a in k:
            var_dict[k.replace(a,b)] = var_dict[k]
            del var_dict[k]

for k in list(var_dict.keys()):
  if 'connect' in k:
    print(k)

model = Mobilenetv2()
x = model.state_dict()

print(set(var_dict.keys()) - set(x.keys()))
print(set(x.keys()) - set(var_dict.keys()))

assert len(set(x.keys()) - set(var_dict.keys())) == 0

for k in list(var_dict.keys()):
  if var_dict[k].ndim == 4:
    if '.conv2' in k:
      var_dict[k] = var_dict[k].transpose((2, 3, 0, 1)).copy(order='C')
    else:
      var_dict[k] = var_dict[k].transpose((3, 2, 0, 1)).copy(order='C')
  if var_dict[k].ndim == 2:
    var_dict[k] = var_dict[k].transpose((1, 0)).copy(order='C')
  if x[k].shape != var_dict[k].shape:
    print(k)
    print(x[k].shape)
    print(var_dict[k].shape)



for k in list(var_dict.keys()):
  var_dict[k] = torch.from_numpy(var_dict[k])
  # print(var_dict[k])


# torch.save(var_dict, 'mobilenetv2.pth')
