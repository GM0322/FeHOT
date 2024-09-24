import torch
from . import CTLayer
from utils import config
import torch.nn.functional as F
from collections import OrderedDict
from torch import fft

class block_net(torch.nn.Module):
    def __init__(self,in_channel,out_channel):
        super(block_net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel,32,kernel_size=(3,11),padding=(1,5))
        self.conv2 = torch.nn.Conv2d(32,32,kernel_size=(3,11),padding=(1,5))
        self.conv3 = torch.nn.Conv2d(32,out_channel,kernel_size=(3,11),padding=(1,5))
        self.relu = torch.nn.PReLU()

    def forward(self,x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        return out


def Fourier_filter(x, threshold_factor, scale):
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).to(x.device)
    threshold = int(H//2*threshold_factor)
    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered

class GroupUNet(torch.nn.Module):
    """ U-Net implementation.

    Based on https://github.com/mateuszbuda/brain-segmentation-pytorch/
    and modified in agreement with their licence:

    -----

    MIT License

    Copyright (c) 2019 mateuszbuda

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_features=32,
        drop_factor=0.0,
        do_center_crop=False,
        num_groups=32,
        filter=False,
        scale=0.5,
        threshold_factor=0.5
    ):
        # set properties of UNet
        super(GroupUNet, self).__init__()
        self.filter=filter
        self.scale=scale
        self.threshold_factor=threshold_factor
        self.do_center_crop = do_center_crop
        kernel_size = 3 if do_center_crop else 2

        self.encoder1 = self._conv_block(
            in_channels,
            base_features,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_1",
        )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._conv_block(
            base_features,
            base_features * 2,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_2",
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._conv_block(
            base_features * 2,
            base_features * 4,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_3",
        )
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._conv_block(
            base_features * 4,
            base_features * 8,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_4",
        )
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._conv_block(
            base_features * 8,
            base_features * 16,
            num_groups,
            drop_factor=drop_factor,
            block_name="bottleneck",
        )

        self.upconv4 = torch.nn.ConvTranspose2d(
            base_features * 16,
            base_features * 8,
            kernel_size=kernel_size,
            stride=2,
        )
        self.decoder4 = self._conv_block(
            base_features * 16,
            base_features * 8,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_4",
        )
        self.upconv3 = torch.nn.ConvTranspose2d(
            base_features * 8,
            base_features * 4,
            kernel_size=kernel_size,
            stride=2,
        )
        self.decoder3 = self._conv_block(
            base_features * 8,
            base_features * 4,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_3",
        )
        self.upconv2 = torch.nn.ConvTranspose2d(
            base_features * 4,
            base_features * 2,
            kernel_size=kernel_size,
            stride=2,
        )
        self.decoder2 = self._conv_block(
            base_features * 4,
            base_features * 2,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_2",
        )
        self.upconv1 = torch.nn.ConvTranspose2d(
            base_features * 2, base_features, kernel_size=kernel_size, stride=2
        )
        self.decoder1 = self._conv_block(
            base_features * 2,
            base_features,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_1",
        )

        self.outconv = torch.nn.Conv2d(
            in_channels=base_features,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x):

        enc1 = self.encoder1(x)

        enc2 = self.encoder2(self.pool1(enc1))

        enc3 = self.encoder3(self.pool2(enc2))

        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = self._center_crop(dec4, enc4.shape[-2], enc4.shape[-1])

        if self.filter:
            enc4 = Fourier_filter(enc4,self.threshold_factor,self.scale)

        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self._center_crop(dec3, enc3.shape[-2], enc3.shape[-1])
        if self.filter:
            enc3 = Fourier_filter(enc3,self.threshold_factor,self.scale)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self._center_crop(dec2, enc2.shape[-2], enc2.shape[-1])
        if self.filter:
            enc2 = Fourier_filter(enc2,self.threshold_factor,self.scale)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self._center_crop(dec1, enc1.shape[-2], enc1.shape[-1])
        if self.filter:
            enc1 = Fourier_filter(enc1,self.threshold_factor,self.scale)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.outconv(dec1)

    def _conv_block(
        self, in_channels, out_channels, num_groups, drop_factor, block_name
    ):
        return torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        block_name + "conv1",
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (
                        block_name + "bn_1",
                        torch.nn.GroupNorm(num_groups, out_channels),
                    ),
                    (block_name + "relu1", torch.nn.ReLU(True)),
                    (block_name + "dr1", torch.nn.Dropout(p=drop_factor)),
                    (
                        block_name + "conv2",
                        torch.nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (
                        block_name + "bn_2",
                        torch.nn.GroupNorm(num_groups, out_channels),
                    ),
                    (block_name + "relu2", torch.nn.ReLU(True)),
                    (block_name + "dr2", torch.nn.Dropout(p=drop_factor)),
                ]
            )
        )

    def _center_crop(self, layer, max_height, max_width):
        if self.do_center_crop:
            _, _, h, w = layer.size()
            xy1 = (w - max_width) // 2
            xy2 = (h - max_height) // 2
            return layer[
                :, :, xy2 : (xy2 + max_height), xy1 : (xy1 + max_width)
            ]
        else:
            return layer

class GradBlock(torch.nn.Module):
    def __init__(self,order=1):
        super().__init__()
        self.order = order
        self.grad1 = torch.nn.Conv2d(1,2,kernel_size=(3,3),padding=(1,1),bias=False)
        self.grad2 = torch.nn.Conv2d(1,3,kernel_size=(3,3),padding=(1,1),bias=False)
        self.resetParameters()
        self.grad1Conv1 = torch.nn.Conv2d(2,8,kernel_size=3,padding=1)
        self.grad1Conv2 = torch.nn.Conv2d(8,8,kernel_size=3,padding=1)
        self.grad1Conv3 = torch.nn.Conv2d(8,2,kernel_size=3,padding=1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        if self.order == 2:
            self.grad2Conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.grad2Conv2 = torch.nn.Conv2d(8, 8, kernel_size=3, padding=1)
            self.grad2Conv3 = torch.nn.Conv2d(8, 3, kernel_size=3, padding=1)

    def forward(self,x):
        x = x[:,0:1,...]
        dx1 = self.grad1(x)
        dx1 = self.grad1Conv1(dx1)**2
        dx1 = self.relu(self.grad1Conv2(dx1))
        dx1 = self.sigmoid(torch.sqrt(torch.abs(self.grad1Conv3(dx1))))
        if(self.order == 2):
            dx2 = self.grad2(x)
            dx2 = self.grad2Conv1(dx2) ** 2
            dx2 = self.relu(self.grad2Conv2(dx2))
            dx2 = self.sigmoid(torch.sqrt(torch.abs(self.grad2Conv3(dx2))))
            dx1 = torch.cat([dx1,dx2],dim=1)
        return dx1

    def resetParameters(self):
        self.grad1.weight.data = torch.nn.Parameter(torch.tensor([[[[0,0.5,0],[0,0,0],[0,-0.5,0]]],[[[0,0,0],[0.5,0,-0.5],[0,0,0]]]]))
        self.grad2.weight.data = torch.nn.Parameter(
            torch.tensor([[[[0, 0.5, 0], [0, -1.0, 0], [0, 0.5, 0]]],
                          [[[0, 0, 0], [0.5, -1.0, 0.5], [0, 0, 0]]],
                          [[[0.25, 0, -0.25], [0, -0.0, 0.], [-0.25, 0, 0.25]]]]))

class primal_dual(torch.nn.Module):
    def __init__(self,n_iter,n_primal,n_dual):
        super(primal_dual, self).__init__()
        self.n_iter = n_iter
        geom, space = CTLayer.getScanParam()
        self.fp = CTLayer.getForwardOperator(geom, space)
        self.bp = CTLayer.getBackWardOperator(geom, space)
        self.layers = torch.nn.ModuleList()
        self.n_primal = n_primal
        self.n_dual = n_dual

        for i in range(n_iter):
            dual_layer = torch.nn.Sequential(
                torch.nn.Conv2d(n_dual+2,32,kernel_size=(3,3),padding=(1,1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(32,32,kernel_size=(3,3),padding=(1,1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(32, n_dual, kernel_size=(3,3),padding=(1,1)),
            )
            primal_layer = torch.nn.Sequential(
                torch.nn.Conv2d(n_primal+1, 32, kernel_size=(3,3),padding=(1,1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(32, 32, kernel_size=(3,3),padding=(1,1)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(32, n_primal, kernel_size=(3,3),padding=(1,1)),
            )
            self.layers.append(dual_layer)
            self.layers.append(primal_layer)

    def forward(self, primal, proj):
        primal = torch.cat([primal] * self.n_primal, dim=1)
        dual = torch.cat([torch.zeros_like(proj,dtype=torch.float32)]*self.n_dual,dim=1).to(proj.device)
        for i in range(self.n_iter):
            evalop = self.fp(primal[:, 1:2, ...])
            update = torch.cat((dual,evalop,proj),dim=1)
            dual = self.layers[2*i](update)

            evalop = self.bp(dual[:, 0:1, ...])
            update = torch.cat([primal, evalop], dim=1)
            update = self.layers[2*i+1](update)
            primal = primal + update
        return primal[:, 0:1, ...]

class hot_net(torch.nn.Module):
    def __init__(self,n_iter,n_order,filter=True,scale=0.5):
        super(hot_net, self).__init__()
        self.n_iter = n_iter
        self.scale = scale
        geom, space = CTLayer.getScanParam()
        self.fp = CTLayer.getForwardOperator(geom, space)
        self.bp = CTLayer.getBackWardOperator(geom, space)
        self.layers = torch.nn.ModuleList()
        self.n_order = n_order
        self.n_channel = 2 if n_order == 1 else 5
        self.relu = torch.nn.ReLU()

        self.layers.append(GradBlock(self.n_order))
        for i in range(n_iter):
            proj_layer = block_net(2,1)
            image_layer = GroupUNet(self.n_channel+2,1,filter=filter,scale=scale)
            self.layers.append(proj_layer)
            self.layers.append(image_layer)

    def forward(self, xinit, proj):
        args = config.getParse()
        # gradBlock = GradBlock(self.n_order).cuda(xinit.device)
        for i in range(self.n_iter):
            proj_new = self.fp(xinit)
            proj_update = torch.cat((proj_new, proj),dim=1)
            proj_update = self.layers[2*i+1](proj_update)

            image_new = self.relu(self.bp(proj_update[:, 0:1, ...]))
            image_update = torch.cat([image_new, xinit], dim=1)
            update_local = image_update[...,args.nSize//2-args.nRSize:args.nSize//2+args.nRSize,args.nSize//2-args.nRSize:args.nSize//2+args.nRSize]
            update_local = torch.cat([update_local,0.01*self.layers[0](update_local)],dim=1)
            update_local = self.relu(self.layers[2*i+2](update_local))
            image_update = F.pad(update_local,(args.nSize//2-args.nRSize,args.nSize//2-args.nRSize,args.nSize//2-args.nRSize,args.nSize//2-args.nRSize))
            xinit = image_new + image_update
        return xinit

    def setScale(self,value):
        for layer in self.layers:
            if(isinstance(layer,GroupUNet)):
                layer.scale = value

def loss_cacl(result,gt, loss):
    args = config.getParse()
    Ix, Iy = torch.meshgrid(torch.arange(-args.nSize/2+0.5, args.nSize/2), torch.arange(-args.nSize/2+0.5, args.nSize/2),indexing='ij')
    mask = (Ix.pow(2) + Iy.pow(2)) > (args.nRSize-3)**2
    result = result.clone()
    result[:,:,mask] = 0
    gt[:,:,mask] = 0
    return loss(result,gt)
