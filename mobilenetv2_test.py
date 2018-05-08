from mobilenetv2 import Mobilenetv2

model = Mobilenetv2()
x = model.state_dict()
# i = 0
# for k in list(x.keys()):
#   if 'layers' in k:
#     print(k)
#     i += 1
# print(i)
for k in list(x.keys()):
  print(k)
