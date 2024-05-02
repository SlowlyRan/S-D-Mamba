from data_provider.data_factory import data_provider
from utils import Args
import matplotlib.pyplot as plt

arg = Args.Args(root_path=r"C:\Repo\pv_data\Data/")
flag = "whole"
data_set, data_loader  = data_provider(arg, flag)

for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,xtt,ytt) in enumerate(data_loader):
#for x,y,xt,yt,xtt,ytt in data_set:
#for x, y, xt, yt in data_set:

    tt = ytt
    break

b= data_set.inverse_transform(a)
print(type(tt))
ttt = data_set.data_stamp_reverse(tt)

print(ttt)
#plt.plot(ttt,b)
#plt.show()
