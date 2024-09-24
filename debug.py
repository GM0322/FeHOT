#
#
# 测试odl的正投反投影算子
from network import CTLayer
import numpy as np
import torch

geom,space = CTLayer.getScanParam()
forwardOperator = CTLayer.getForwardOperator(geom,space)
backwardOperator = CTLayer.getBackWardOperator(geom,space)
img = np.fromfile(r'D:\GM\dataset\HOT_TV\abdomen\train\label\L004_000.raw',dtype=np.float32).reshape(1,1,512,512)
img = torch.from_numpy(img[:,:, ::-1, :].copy()).permute(0,1,3,2).cuda().contiguous()
# p = forwardOperator(img)
p = torch.from_numpy(np.fromfile(r'D:\GM\dataset\HOT_TV\abdomen\train\proj\L004_000.raw',dtype=np.float32)).view(1,1,360,300).cuda()

rec = backwardOperator(p)

import pylab
pylab.figure(1)
pylab.imshow(img[0,0,...].cpu(),cmap='gray')
pylab.figure(2)
pylab.subplot(121)
pylab.imshow(p[0,0,...].cpu(),cmap='gray')
pylab.figure(3)
pylab.imshow(rec[0,0,...].data.cpu(),cmap='gray')

#
# 测试生成数据
# from utils import config
# from utils import create_data
# import numpy as np
#
#
# args = config.getParse()
# gendata = create_data.DatasetGenerate(args)
# gendata.generate()

# img = np.fromfile(r'E:\dataset\HOT_TV\abdomen\train\label\L004_000.raw',dtype=np.float32).reshape(512,512)
# p,pn,fbp,fbpn,sirt,sirtn = gendata.projection(img)
#
# rec = backwardOperator(p)
# import pylab
# pylab.figure(4)
# pylab.subplot(121)
# pylab.imshow(p,cmap='gray')
# pylab.subplot(122)
# pylab.imshow(pn,cmap='gray')
# pylab.figure(5)
# pylab.subplot(221)
# pylab.imshow(fbp.T,cmap='gray',vmin=0.0,vmax=0.43)
# pylab.subplot(222)
# pylab.imshow(fbpn.T,cmap='gray',vmin=0.0,vmax=0.43)
# pylab.subplot(223)
# pylab.imshow(sirt.T,cmap='gray',vmin=0.0,vmax=0.43)
# pylab.subplot(224)
# pylab.imshow(sirtn.T,cmap='gray',vmin=0.0,vmax=0.43)
#
# pylab.figure(6)
# pylab.imshow(rec)

# 读取数据
# from dataset import dataset
# from utils import config
# args = config.getParse()
# trainData = dataset.ProjAndImageDataset(args.trainData,args.nViews,args.nBins,args.nSize)
# import torch
# from dataset import dataset
# from network import network
# from utils import config
# import time
# args = config.getParse()
# trainData = dataset.ProjAndImageDataloader(args.trainData+args.isNoisy,args.nViews,args.nBins,args.nSize,1,False)
# model = network.chambolle_pock(args.iter)
# model.layers = torch.load('./checkpoint/chambolle_pock_abdomen.pt')
# for step,(x,y,fileName) in enumerate(trainData):
#     if step<1:
#         primal = torch.zeros_like(y, dtype=torch.float32).cuda()
#         start = time.time()
#         out = model(primal, x.cuda())
#         torch.cuda.synchronize()
#         end = time.time()
#         with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
#                 outputs = model(primal, x.cuda())
#         print(prof.table())
# import pylab
# pylab.imshow(out[0,0,...].data.cpu().numpy(),cmap='gray')



# FFT
# import numpy as np
# from pypher.pypher import psf2otf
# import PIL.Image as Image
# from torch import fft
# import torch
#
# def L0Smoothing(Im, lambd = 2e-2, kappa = 2.0):
# 	# L0 Smoothing
# 	# Input:
# 	#   Im: Input UINT8 image, both grayscale and color images are acceptable.
# 	#   lambd: Smoothing parameter controlling the degree of smooth. (See [1])
# 	#          Typically it is within the range [1e-3, 1e-1], 2e-2 by default.
# 	#   kappa: Parameter that controls the rate. (See [1])
# 	#          Small kappa results in more iteratioins and with sharper edges.
# 	#          We select kappa in (1, 2].
# 	#          kappa = 2 is suggested for natural images.
#
# 	# Example:
# 	#   Im = imread('test.png')
# 	#   S = L0Smoothing(Im, 2e-2, 2.0)
# 	#   imshow(S)
#
# 	S = Im.astype(np.float64) / 255.0
# 	betamax = 1e5
# 	fx = np.array([[-1, 1]])
# 	fy = np.array([[-1], [1]])
# 	N, M, D = S.shape
# 	sizeI2D = np.array([N, M])
# 	otfFx = psf2otf(fx, sizeI2D)
# 	otfFy = psf2otf(fy, sizeI2D)
# 	Normin1 = np.fft.fft2(S.T).T
# 	Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2
# 	if D > 1:
# 		D2 = np.zeros((N, M, D), dtype=np.double)
# 		for i in range(D):
# 			D2[:, :, i] = Denormin2
# 		Denormin2 = D2
# 	beta = lambd * 2
# 	while beta < betamax:
# 		Denormin = 1 + beta * Denormin2
# 		# Referenced from L-Dreams's blog
# 		# h-v subproblem
# 		h1 = np.diff(S, 1, 1)
# 		h2 = np.reshape(S[:, 0], (N, 1, 3)) - np.reshape(S[:, -1], (N, 1, 3))
# 		h = np.hstack((h1, h2))
# 		v1 = np.diff(S, 1, 0)
# 		v2 = np.reshape(S[0, :], (1, M, 3)) - np.reshape(S[-1, :], (1, M, 3))
# 		v = np.vstack((v1, v2))
# 		if D == 1:
# 			t = (h ** 2 + v ** 2) < lambd / beta
# 		else:
# 			t = np.sum((h ** 2 + v ** 2), 2) < lambd / beta
# 			t1 = np.zeros((N, M, D), dtype=bool)
# 			for i in range(D):
# 				t1[:, :, i] = t
# 			t = t1
# 		h[t] = 0
# 		v[t] = 0
# 		# S subproblem
# 		Normin2 = np.hstack((np.reshape(h[:, -1], (N, 1, 3)) - np.reshape(h[:, 0], (N, 1, 3)), -np.diff(h, 1, 1)))
# 		Normin2 = Normin2 + np.vstack(
# 			(np.reshape(v[-1, :], (1, M, 3)) - np.reshape(v[0, :], (1, M, 3)), -np.diff(v, 1, 0)))
# 		FS = (Normin1 + beta * np.fft.fft2(Normin2.T).T) / Denormin
# 		S = np.real(np.fft.ifft2(FS.T).T)
# 		beta *= kappa
# 	return S
#
#
# def Fourier_filter(x, threshold_factor, scale):
# 	x_freq = fft.fftn(x, dim=(-2, -1))
# 	x_freq = fft.fftshift(x_freq, dim=(-2, -1))
#
# 	B, C, H, W = x_freq.shape
# 	mask = torch.ones((B, C, H, W)).to(x.device)
# 	threshold = int(H//2*threshold_factor)
# 	crow, ccol = H // 2, W // 2
# 	mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
# 	x_freq = x_freq * mask
#
# 	# IFFT
# 	x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
# 	x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
#
# 	return x_filtered
#
# import torchvision
# im = np.array(Image.open(r'C:\ProgramData\opencv\sources\samples\data\lena.jpg'))
# print("Image Loaded.")
# #L0 Smoothing
# print("Image Processing.")
# S = L0Smoothing(im, 8e-3, 2.0)
# #save image
# print("Image Saving.")
# S1 = Image.fromarray(np.uint8(S*255))
# im_t = torch.from_numpy(im).float().permute(2,0,1)[None,...]/255.0
# im_t = im_t+0.03*torch.randn_like(im_t)
#
# im_t = im_t[:,1:2,...]
# S = torch.from_numpy(S).float().permute(2,0,1)[None,1:2,...]
# ff = []
# for i in range(5):
# 	f = Fourier_filter(im_t, 0.5, i*0.4)
# 	ff.append(f)
# fs = []
# for i in range(5):
# 	f = Fourier_filter(im_t, 0.2*i, 1.0)
# 	fs.append(f)
#
# import matplotlib.pyplot as plt
# # ff = torch.cat(ff,dim=0)
# # sff = (S+ff)/(S+ff).max()
# # ff = ff/ff.max()
# # fs = torch.cat(fs,dim=0)
# # sfs = (S+fs)/(S+fs).max()
# # fs = fs/fs.max()
# # [:,:,50:150,100:200],cmap='gray',cmap='gray'
# plt.figure(1)
# plt.imshow(torchvision.utils.make_grid(torch.cat([f/f.max() for f in ff],dim=0),nrow=5,padding=2)[0],cmap='gray',vmax=1)
# plt.figure(2)
# plt.imshow(torchvision.utils.make_grid(torch.cat([f/f.max() for f in fs],dim=0),nrow=5,padding=2)[0],cmap='gray',vmax=1)
# plt.show()
# #
# # f = Fourier_filter(im_t,0.5,0.0)[0,...].permute(1,2,0)
# # sf = f+S
# # S2 = Image.fromarray(np.uint8(sf*255))
# # plt.figure(1)
# # plt.imshow(im[:,:,1],cmap='gray')
# # plt.figure(2)
# # plt.imshow(sf[:,:,1],cmap='gray')
# # plt.figure(3)
# # plt.imshow(S[:,:,1],cmap='gray')
# # plt.figure(4)
# # plt.imshow(f[:,:,1],cmap='gray')
# # plt.show()
# #
# #
# # # S1.save('pflower_L0Smoothing.jpg')
# # # print("Done.")

# import numpy as np
# I1 = np.load(r'C:\baidunetdiskdownload\Phantom\Phantom_batch1.npy')
# I2 = np.load(r'C:\baidunetdiskdownload\Phantom\Phantom_batch2.npy')
# I3 = np.load(r'C:\baidunetdiskdownload\Phantom\Phantom_batch3.npy')
# I4 = np.load(r'C:\baidunetdiskdownload\Phantom\Phantom_batch4.npy')
# for i in range(I1.shape[0]):
# 	I1[i,...].tofile(r'D:\GM0322\dataset\HOT_TV\AAPM\train\label\phantom1_{:03d}.raw'.format(i))
# 	I2[i,...].tofile(r'D:\GM0322\dataset\HOT_TV\AAPM\train\label\phantom2_{:03d}.raw'.format(i))
# 	I3[i,...].tofile(r'D:\GM0322\dataset\HOT_TV\AAPM\train\label\phantom3_{:03d}.raw'.format(i))
# 	I4[i,...].tofile(r'D:\GM0322\dataset\HOT_TV\AAPM\val\label\phantom4_{:03d}.raw'.format(i))



