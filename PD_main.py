import os
import numpy as np
import torch
from utils import config
from utils import create_data
from dataset import dataset
from network import network
import time
from tqdm import tqdm

args = config.getParse()
args.trainData = r'D:\GM\dataset\HOT_TV\AAPM\train\proj'
args.isNoisy='_noisy'#
args.iter = 10
args.batchSize = 8
args.epoch = 300
trainData = dataset.ProjAndImageDataloader(args.trainData+args.isNoisy,args.nViews,args.nBins,args.nSize,args.batchSize,True)
model = network.primal_dual(args.iter,n_primal=5,n_dual=5).to(args.gpuDevice[0])
criterion = torch.nn.MSELoss(reduction='sum')
optim = torch.optim.Adam(model.parameters(), lr=args.learnRate)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[i * args.epoch // 5 for i in range(1, 5)], gamma=0.75)
print('---------------------------start train ' + args.model+'-----------------------------------------')
print(args)
start = time.time()
losses = []
# model.layers = torch.load('./checkpoint/PD' + args.isNoisy + '.pt', map_location='cuda:{}'.format(args.gpuDevice[0]))

for epoch in range(args.epoch):
    loop = tqdm(enumerate(trainData),total=len(trainData))
    for step,(x,y,_) in loop:
        xinit = torch.zeros_like(y,dtype=torch.float32).cuda(args.gpuDevice[0])
        out = model(xinit,x.cuda(args.gpuDevice[0]))
        lossV = network.loss_cacl(out,y.cuda(args.gpuDevice[0]),criterion)
        optim.zero_grad()
        lossV.backward()
        optim.step()
        if(step%100 == 0):
            out.data.cpu().numpy().tofile('temp/PD'+args.isNoisy+'rec.raw')
            y.data.cpu().numpy().tofile('temp/PD'+args.isNoisy+'lab.raw')
        loop.set_description('epoch={},step = {}'.format(epoch, step))
        loop.set_postfix(loss=lossV.item())
        losses.append(lossV.item())
    curTime = time.time()
    print('epoch={},training time:{}'.format(epoch,curTime-start))
    lr_scheduler.step()
    torch.save(model.layers,'./checkpoint/PD'+args.isNoisy+'.pt')

print('---------------------------start save model and test data-----------------------------------------')
model.eval()
loss_np = np.array([loss for loss in losses])
loss_np.tofile('./checkpoint/PD'+args.isNoisy+'_lossupdate.raw')
torch.save(model.layers,'./checkpoint/PD'+args.isNoisy+'.pt')

# model.layers = torch.load('./checkpoint/PD'+args.isNoisy+'.pt',map_location='cuda:{}'.format(args.gpuDevice[0]))
trainData = dataset.ProjAndImageDataloader(args.trainData+args.isNoisy,args.nViews,args.nBins,args.nSize,1,False)
args.valData = r'D:\GM0322\dataset\HOT_TV\AAPM\val\proj'
valData = dataset.ProjAndImageDataloader(args.valData+args.isNoisy,args.nViews,args.nBins,args.nSize,1,False)

if(os.path.isdir(args.trainData+'/../PD'+args.isNoisy) == False):
    os.mkdir(args.trainData+'/../PD'+args.isNoisy)
if(os.path.isdir(args.valData+'/../PD'+args.isNoisy) == False):
    os.mkdir(args.valData+'/../PD'+args.isNoisy)
for step,(x,y,fileName) in tqdm(enumerate(trainData),total=len(trainData)):
    xinit = torch.zeros_like(y, dtype=torch.float32).cuda(args.gpuDevice[0])
    out = model(xinit, x.cuda(args.gpuDevice[0]))
    out.data.cpu().numpy().tofile(args.trainData+'/../PD'+args.isNoisy+'/'+fileName[0])
for step,(x,y,fileName) in tqdm(enumerate(valData),total=len(valData)):
    xinit = torch.zeros_like(y, dtype=torch.float32).cuda(args.gpuDevice[0])
    out = model(xinit, x.cuda(args.gpuDevice[0]))
    out.data.cpu().numpy().tofile(args.valData+'/../PD'+args.isNoisy+'/'+fileName[0])
print('---------------------------all prcessing finished, please checking.-----------------------------------------')

