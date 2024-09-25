import os

import numpy as np
import torch
from utils import config
from utils import create_data
from dataset import dataset
from network import network
import time
from tqdm import tqdm
import torchvision
from network import CTLayer
import astra



args = config.getParse()
model = eval("network."+args.model)(args.iter,args.order,filter=args.filter).cuda(args.gpuDevice[0])
savePath = args.model + str(args.order)
if args.filter == False:
    savePath += '_nf'
model.layers = torch.load('./checkpoint/'+args.dataType[0]+'_'+savePath+'.pt',map_location='cuda:{}'.format(args.gpuDevice[0]))
# model.scale = 0.5

xinit = torch.zeros(size=(1,1,args.nSize,args.nSize), dtype=torch.float32).cuda(args.gpuDevice[0])
x = torch.from_numpy(np.fromfile(r'D:\GM\dataset\HOT_TV\abdomen\val\proj\L175_100.raw',     #120
                                                                                               dtype=np.float32)).view(1,1,args.nViews,args.nBins).cuda(args.gpuDevice[0])
gt = torch.from_numpy(np.fromfile(r'D:\GM\dataset\HOT_TV\abdomen\val\label\L175_100.raw',
                                 dtype=np.float32)).view(1,1,args.nSize,args.nSize).cuda(args.gpuDevice[0]).permute(0,1,3,2).flip(dims=(3,))
res = [gt]
for i in range(11):
    # model.setScale(0.1*i)
    out = model(xinit, x.cuda(args.gpuDevice[0]))
    res.append(out)
res = torch.cat(res,0)[:,:,256-78:256+78,256-78:256+78]
import matplotlib.pyplot as plt
# plt.figure(figsize=(20,4))
# plt.imshow(torchvision.utils.make_grid(res.cpu(),nrow=6)[0],cmap='gray',vmin=0.0,vmax=0.3)
plt.figure(1)
plt.imshow(res[0,0,...].data.cpu().numpy(),cmap='gray',vmin=0.0,vmax=0.2)

plt.figure(2)
plt.imshow(res[1,0,...].data.cpu().numpy(),cmap='gray',vmin=0.0,vmax=0.3)
plt.figure(3)
plt.imshow(rec_fbp[256-78:256+78,256-78:256+78],cmap='gray',vmin=0.0,vmax=0.2)
plt.show()
