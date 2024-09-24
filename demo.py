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


geom,space = CTLayer.getScanParam()
forwardOperator = CTLayer.getForwardOperator(geom,space)
backwardOperator = CTLayer.getBackWardOperator(geom,space)
img = np.fromfile(r'D:\jiangxu\20240111-humanBone\head of femur\1-2\FDKrecon_low_2048x2048x512.raw',     #120
                                 dtype=np.float32,offset=2048*2048*48*4,count=2048*2048).reshape(1,1,2048,2048)
img = 2*torch.from_numpy(img[:,:, ::4, ::4].copy()).permute(0,1,3,2).cuda(args.gpuDevice[0]).contiguous()
x = forwardOperator(img).detach()
gt = img

vol_geom = astra.create_vol_geom(args.nSize, args.nSize, -args.nSize / 2.0 * args.fPixelSize,
                                      args.nSize / 2.0 * args.fPixelSize,
                                      -args.nSize / 2.0 * args.fPixelSize, args.nSize / 2.0 * args.fPixelSize)
proj_geom = astra.create_proj_geom('fanflat', args.fCellSize, args.nBins,
                                   np.linspace(0, 2 * np.pi, args.nViews, False), args.fSod, args.fOdd)
rec_id = astra.data2d.create('-vol', vol_geom)
proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
sin_id, sinogram = astra.create_sino(img[0,0,...].data.cpu().numpy(),proj_id)
cfg = astra.astra_dict('FBP_CUDA')
cfg['option'] = {'FilterType': 'shepp-logan'}
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sin_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)
astra.algorithm.delete(alg_id)
rec_fbp = astra.data2d.get(rec_id)


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