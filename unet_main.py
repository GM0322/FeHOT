import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import time
from network import network

class roidata(Dataset):
    def __init__(self,path,nSize=512,nRSize=80):
        self.path = path
        self.nSize = nSize
        self.nRSize = nRSize
        self.files = os.listdir(path)
        self.input = torch.zeros(size=(len(self.files),1,2*nRSize,2*nRSize),dtype=torch.float32)
        self.target = torch.zeros(size=(len(self.files),1,2*nRSize,2*nRSize),dtype=torch.float32)
        self.loadData()

    def loadData(self):
        lindex = self.nSize//2-self.nRSize
        rindex = self.nSize//2+self.nRSize
        loop = tqdm(enumerate(self.files),total=len(self.files))
        for i, file in loop:
            input = torch.from_numpy(np.fromfile(self.path+'/'+file,dtype=np.float32)).view(1,self.nSize,self.nSize)
            target = torch.from_numpy(np.fromfile(self.path+'/../label/'+file,dtype=np.float32)).view(1,self.nSize,self.nSize)
            self.input[i,...] = input[:,lindex:rindex,lindex:rindex]
            self.target[i,...] = target[:,lindex:rindex,lindex:rindex]

    def __getitem__(self, item):
        return self.input[item,...],self.target[item,...],self.files[item]

    def __len__(self):
        return len(self.files)

def roiDataLoader(path,nSize,nRSize,**kwargs):
    dataset = roidata(path=path,nSize=nSize,nRSize=nRSize)
    return DataLoader(dataset,batch_size=kwargs['batch_size'],shuffle=kwargs['shuffle'])


""" Parts of the U-Net model """

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down_Block(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, drop=0.5):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.down = nn.Sequential(nn.MaxPool2d(2), nn.Dropout(drop))

    def forward(self, x):
        c = self.conv(x)
        return c, self.down(c)


class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels, drop):
        super().__init__()
        self.conv = nn.Sequential(
            DoubleConv(in_channels, out_channels), nn.Dropout(drop)
        )

    def forward(self, x):
        return self.conv(x)


class Up_Block(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, drop=0.5, attention=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2)
        )
        self.conv = nn.Sequential(
            DoubleConv(in_channels, out_channels), nn.Dropout(p=drop)
        )
        self.attention = attention
        if attention:
            self.gating = GatingSignal(in_channels, out_channels)
            self.att_gat = Attention_Gate(out_channels)

    def forward(self, x, conc):
        x1 = self.up(x)
        if self.attention:
            gat = self.gating(x)
            map, att = self.att_gat(conc, gat)
            x = torch.cat([x1, att], dim=1)
            return map, self.conv(x)
        else:
            x = torch.cat([conc, x1], dim=1)
            return None, self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class GatingSignal(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super(GatingSignal, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        return self.activation(x)


class Attention_Gate(nn.Module):
    def __init__(self, in_channels):
        super(Attention_Gate, self).__init__()
        self.conv_theta_x = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, 1), stride=(2, 2)
        )
        self.conv_phi_g = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.att = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=(1, 1)),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, x, gat):
        theta_x = self.conv_theta_x(x)
        phi_g = self.conv_phi_g(gat)
        res = torch.add(phi_g, theta_x)
        res = self.att(res)
        # print(res.size(), x.size())
        return res, torch.mul(res, x)



class Unet(nn.Module):
    def __init__(self, filters, drop_r=0.5, attention=True):
        super(Unet, self).__init__()
        self.down1 = Down_Block(1, filters)
        self.down2 = Down_Block(filters, filters * 2, drop_r)
        self.down3 = Down_Block(filters * 2, filters * 4, drop_r)
        self.down4 = Down_Block(filters * 4, filters * 8, drop_r)

        self.bridge = Bridge(filters * 8, filters * 16, drop_r)

        self.up1 = Up_Block(filters * 16, filters * 8, drop_r, attention)
        self.up2 = Up_Block(filters * 8, filters * 4, drop_r, attention)
        self.up3 = Up_Block(filters * 4, filters * 2, drop_r, attention)
        self.up4 = Up_Block(filters * 2, filters, drop_r, attention)

        self.outc = OutConv(filters, 1)

    def forward(self, x):
        c1, x1 = self.down1(x)
        c2, x2 = self.down2(x1)
        c3, x3 = self.down3(x2)
        c4, x4 = self.down4(x3)
        bridge = self.bridge(x4)
        _, x = self.up1(bridge, c4)
        _, x = self.up2(x, c3)
        att, x = self.up3(x, c2)
        _, x = self.up4(x, c1)
        mask = self.outc(x)
        return mask

def loss_cacl(result,gt, loss,RSize=112):
    Ix, Iy = torch.meshgrid(torch.arange(-RSize+0.5, RSize), torch.arange(-RSize+0.5, RSize),indexing='ij')
    mask = (Ix.pow(2) + Iy.pow(2)) > (RSize-3)**2
    result = result.clone()
    gt = gt[:]
    result[:,:,mask] = 0
    gt[:,:,mask] = 0
    return loss(result,gt)

noisy = '_noisy'#''#
path = r'D:\GM\dataset\HOT_TV\AAPM\train\FBP'+noisy
RSize = 112
dataloader = roiDataLoader(path,512,RSize,batch_size=16,shuffle=True)

device = 'cuda:2'
model = Unet(32,drop_r=0.0).to(device)
optim = torch.optim.Adam(model.parameters(),lr=1e-5)
criterion = torch.nn.MSELoss(reduction='sum')
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[i * 100 // 5 for i in range(1, 5)], gamma=0.75)
losses = []
start = time.time()
for epoch in range(400):
    loop = tqdm(enumerate(dataloader),total=len(dataloader))
    for step,(x,y,file) in loop:
        out = model(x.to(device))
        optim.zero_grad()
        # loss = criterion(out,y.to(device))
        loss = loss_cacl(out, y.to(device), criterion,RSize)
        loss.backward()
        optim.step()
        if(step%100 == 0):
            out.data.cpu().numpy().tofile('temp/unet_rec.raw')
            y.data.cpu().numpy().tofile('temp/unet_lab.raw')
        loop.set_description('epoch={},step = {}'.format(epoch, step))
        loop.set_postfix(loss=loss.item())
        losses.append(loss.item())
    curTime = time.time()
    lr_scheduler.step()
    print('epoch={},training time:{}'.format(epoch,curTime-start))

print('---------------------------start save model and test data-----------------------------------------')
model.eval()
loss_np = np.array([loss for loss in losses])
loss_np.tofile('./checkpoint/unet'+noisy+'_lossupdate.raw')
torch.save(model,'./checkpoint/unet'+noisy+'.pt')

trainloader = roiDataLoader(path,512,112,batch_size=1,shuffle=False)
if(os.path.isdir(path+'/../unet'+noisy) == False):
    os.mkdir(path+'/../unet'+noisy)
for step,(x,y,fileName) in tqdm(enumerate(trainloader),total=len(trainloader)):
    out = model(x.to(device))
    out.data.cpu().numpy().tofile(path+'/../unet'+noisy+'/'+fileName[0])
path = r'D:\GM0322\dataset\HOT_TV\AAPM\val\FBP'+noisy
valloader = roiDataLoader(path,512,112,batch_size=1,shuffle=False)
if(os.path.isdir(path+'/../unet'+noisy) == False):
    os.mkdir(path+'/../unet'+noisy)
for step,(x,y,fileName) in tqdm(enumerate(valloader),total=len(valloader)):
    out = model(x.to(device))
    out.data.cpu().numpy().tofile(path+'/../unet'+noisy+'/'+fileName[0])