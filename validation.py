import numpy as np
import torch
from network import network
from utils import config

args = config.getParse()
model = network.chambolle_pock(args.iter)
model.layers = torch.load('./checkpoint/'+args.model+'_'+args.dataType[0]+args.isNoisy+'.pt')
input = torch.from_numpy(np.fromfile(r'E:\dataset\HOT_TV\abdomen\val\proj_noisy\L107_000.raw',
                                     dtype=np.float32).reshape(1,1,args.nViews,args.nBins))
primal = torch.zeros((1,1,args.nSize,args.nSize), dtype=torch.float32).cuda()
out = model(primal, input.cuda())
import pylab
pylab.imshow(out[0,0,...].data.cpu().numpy(),cmap='gray')