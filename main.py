import os

import numpy as np
import torch
from utils import config
from utils import create_data
from dataset import dataset
from network import network
import time
from tqdm import tqdm

def train():
    args = config.getParse()
    if args.isCreateData == True:
        print('---------------------------start create data-----------------------------------------')
        genData = create_data.DatasetGenerate(args)
        genData.generate()

    trainData = dataset.ProjAndImageDataloader(args.trainData+args.isNoisy,args.nViews,args.nBins,args.nSize,args.batchSize,True)

    model = eval("network."+args.model)(args.iter,args.order,filter=args.filter).cuda(args.gpuDevice[0])
    loss = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.Adam(model.parameters(), lr=args.learnRate)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[i * args.epoch // 5 for i in range(1, 5)], gamma=0.75)
    # scheduer = torch.optim.lr_scheduler.StepLR(optim, gamma=0.5, step_size=100)
    print('---------------------------start train ' + args.model+'-----------------------------------------')
    print(args)
    start = time.time()
    losses = []
    savePath = args.model + str(args.order)
    if args.filter == False:
        savePath += '_nf'
    for epoch in range(args.epoch):
        loop = tqdm(enumerate(trainData),total=len(trainData))
        for step,(x,y,_) in loop:
            # y = y.permute(0,1,3,2)
            xinit = torch.zeros_like(y,dtype=torch.float32).cuda(args.gpuDevice[0])
            out = model(xinit,x.cuda(args.gpuDevice[0]))
            lossV = network.loss_cacl(out,y.cuda(args.gpuDevice[0]),loss)
            optim.zero_grad()
            lossV.backward()
            optim.step()
            if(step%100 == 0):
                out.data.cpu().numpy().tofile('temp/'+savePath+args.isNoisy+'rec.raw')
                y.data.cpu().numpy().tofile('temp/'+savePath+args.isNoisy+'lab.raw')
            loop.set_description('epoch={},step = {}'.format(epoch, step))
            loop.set_postfix(loss=lossV.item())
            losses.append(lossV.item())
        curTime = time.time()
        print('epoch={},training time:{}'.format(epoch,curTime-start))
        lr_scheduler.step()
        torch.save(model.layers,'./checkpoint/'+args.dataType[0]+'_'+savePath+'.pt')

    print('---------------------------' + args.model+' model train finish-----------------------------------------')
    print('---------------------------start save model and test data-----------------------------------------')
    model.eval()
    loss_np = np.array([loss for loss in losses])
    loss_np.tofile('./checkpoint/'+args.dataType[0]+'_'+savePath+'_lossupdate.raw')
    torch.save(model.layers,'./checkpoint/'+args.dataType[0]+'_'+savePath+'.pt')

    model.layers = torch.load('./checkpoint/'+args.dataType[0]+'_'+savePath+'.pt', map_location='cuda:{}'.format(args.gpuDevice[0]))


    trainData = dataset.ProjAndImageDataloader(args.trainData+args.isNoisy,args.nViews,args.nBins,args.nSize,1,False)
    valData = dataset.ProjAndImageDataloader(args.valData+args.isNoisy,args.nViews,args.nBins,args.nSize,1,False)

    if(os.path.isdir(args.trainData+'/../'+savePath+args.isNoisy) == False):
        os.mkdir(args.trainData+'/../'+savePath+args.isNoisy)
    if(os.path.isdir(args.valData+'/../'+savePath+args.isNoisy) == False):
        os.mkdir(args.valData+'/../'+savePath+args.isNoisy)
    for step,(x,y,fileName) in tqdm(enumerate(trainData),total=len(trainData)):
        xinit = torch.zeros_like(y, dtype=torch.float32).cuda(args.gpuDevice[0])
        out = model(xinit, x.cuda(args.gpuDevice[0]))
        out.data.cpu().numpy().tofile(args.trainData+'/../'+savePath+args.isNoisy+'/'+fileName[0])
    for step,(x,y,fileName) in tqdm(enumerate(valData),total=len(valData)):
        xinit = torch.zeros_like(y, dtype=torch.float32).cuda(args.gpuDevice[0])
        out = model(xinit, x.cuda(args.gpuDevice[0]))
        out.data.cpu().numpy().tofile(args.valData+'/../'+savePath+args.isNoisy+'/'+fileName[0])
    print('---------------------------all prcessing finished, please checking.-----------------------------------------')

if __name__ == '__main__':
    train()
