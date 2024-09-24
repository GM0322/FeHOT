import  argparse
import numpy as np

def getParse():
    args = argparse.ArgumentParser()
    # scan param
    args.add_argument('--scanType',type=str,default='FanBeam')
    args.add_argument('--nViews',type=int,default=360)
    args.add_argument('--nBins',type=int,default=300)
    args.add_argument('--nSize',type=int,default=512)
    args.add_argument('--nRSize',type=int,default=80)
    args.add_argument('--fSod',type=float,default=500.0)
    args.add_argument('--fOdd',type=float,default=500.0)
    args.add_argument('--fCellSize',type=float,default=1.1)
    args.add_argument('--noisyLevel',type=float,default=5e4)

    # generate data param
    args.add_argument('--isCreateData',type=bool,default=False)
    args.add_argument('--dataPath',type=str,default=r'D:\GM\dataset\HOT_TV')
    args.add_argument('--dataType',type=list,default=['abdomen'])#,'chest'abdomen
    args.add_argument('--dataFile',type=list,default=['proj','label','FBP','SIRT'])

    # network param
    args.add_argument('--model',type=str,default='hot_net')
    args.add_argument('--iter',type=int,default=5)
    args.add_argument('--trainData',type=str,default=r'D:\GM\dataset\HOT_TV\abdomen\train\proj')
    args.add_argument('--valData',type=str,default=r'D:\GM\dataset\HOT_TV\abdomen\val\proj')
    args.add_argument('--filter',type=bool,default=True)
    args.add_argument('--order',type=int,default=1)
    args.add_argument('--isNoisy',type=str,default='')#_noisy
    # args.add_argument('--savePath',type=str,default=r'abdomen')
    args.add_argument('--gpuDevice', type=list, default=[2])
    args.add_argument('--learnRate',type=float,default=1e-5)
    args.add_argument('--epoch',type=int,default=300)
    args.add_argument('--batchSize',type=int,default=16)
    param = args.parse_args()
    args.add_argument('--fPixelSize',type=float,default=getPixelSize(param))
    return args.parse_args()

def getPixelSize(args):
    halfDet = args.fCellSize*args.nBins/2.0
    length = np.sqrt((args.fSod+args.fOdd)**2+halfDet**2)
    return args.fSod*halfDet/(length*args.nRSize).astype(np.float32).tolist()

