import torch
import  numpy as np
import os
import astra

class DatasetGenerate():
    def __init__(self,args):
        self.args = args
        self.vol_geom = astra.create_vol_geom(args.nSize,args.nSize,-args.nSize/2.0*args.fPixelSize,args.nSize/2.0*args.fPixelSize,
                                              -args.nSize/2.0*args.fPixelSize,args.nSize/2.0*args.fPixelSize)
        self.scanType = args.scanType
        self.proj_geom = None
        if self.scanType == 'FanBeam':
            self.proj_geom = astra.create_proj_geom('fanflat', args.fCellSize, args.nBins,np.linspace(0, 2 * np.pi, args.nViews, False), args.fSod, args.fOdd)
        elif self.scanType == 'Parallel':
            self.proj_geom = astra.create_proj_geom('parallel',args.fCellSize,args.nBins,np.linspace(0, 2 * np.pi, args.nViews, False))
        self.proj_id = astra.create_projector('cuda',self.proj_geom,self.vol_geom)

    def projection(self,ImageData,isReon=False):
        sin_id, sinogram = astra.create_sino(ImageData,self.proj_id)
        maxv = sinogram.max()
        counts = self.args.noisyLevel*np.exp(-sinogram/maxv)
        noisy_counts = np.random.poisson(counts)
        noisy_sinogram = -np.log(noisy_counts/self.args.noisyLevel)*maxv
        noisy_sinogram = noisy_sinogram.astype(np.float32)
        # recon
        rec_fbp = None
        rec_sirt = None
        rec_fbp_noisy = None
        rec_sirt_noisy = None
        if isReon == True:
            rec_id = astra.data2d.create('-vol',self.vol_geom)
            cfg = astra.astra_dict('SIRT_CUDA')
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = sin_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id,100)
            astra.algorithm.delete(alg_id)
            rec_sirt = astra.data2d.get(rec_id)
            cfg['type'] = 'FBP_CUDA'
            cfg['option'] = {'FilterType': 'shepp-logan'}
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            astra.algorithm.delete(alg_id)
            rec_fbp = astra.data2d.get(rec_id)

            astra.data2d.store(sin_id,noisy_sinogram)
            cfg = astra.astra_dict('SIRT_CUDA')
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = sin_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id,100)
            astra.algorithm.delete(alg_id)
            rec_sirt_noisy = astra.data2d.get(rec_id)
            cfg['type'] = 'FBP_CUDA'
            cfg['option'] = {'FilterType': 'shepp-logan'}
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            rec_fbp_noisy= astra.data2d.get(rec_id)
            astra.algorithm.delete(alg_id)
            # astra.data2d.delete(rec_id)
            astra.data2d.delete(sin_id)
            astra.projector.delete(rec_id)
        return sinogram,noisy_sinogram,rec_fbp,rec_sirt,rec_fbp_noisy,rec_sirt_noisy

    def generate(self):
        index = 0
        for _, dataType in enumerate(self.args.dataType):
            for _,folder in enumerate(self.args.dataFile):
                if(os.path.isdir(self.args.dataPath+'/'+dataType+'/train/'+folder+'_noisy' ) == False):
                    os.mkdir(self.args.dataPath+'/'+dataType+'/train/'+folder+'_noisy' )
                if (os.path.isdir(self.args.dataPath + '/' + dataType + '/train/' + folder) == False):
                    os.mkdir(self.args.dataPath + '/' + dataType + '/train/' + folder)
                if(os.path.isdir(self.args.dataPath+'/'+dataType+'/val/'+folder+'_noisy' ) == False):
                    os.mkdir(self.args.dataPath+'/'+dataType+'/val/'+folder+'_noisy' )
                if (os.path.isdir(self.args.dataPath + '/' + dataType + '/val/' + folder) == False):
                    os.mkdir(self.args.dataPath + '/' + dataType + '/val/' + folder)
            print('---------------------------start generate train data-----------------------------------------')
            files = os.listdir(self.args.dataPath+'/'+dataType+'/train/label')
            for step,file in enumerate(files):

                img = np.fromfile(self.args.dataPath+'/'+dataType+'/train/label/'+file,dtype=np.float32).reshape(self.args.nSize,self.args.nSize)
                sinogram, noisy_sinogram, rec_fbp, rec_sirt, rec_fbp_noisy, rec_sirt_noisy = self.projection(img,True)
                sinogram.tofile(self.args.dataPath + '/' + dataType + '/train/' + self.args.dataFile[0]+'/'+file)
                noisy_sinogram.tofile(self.args.dataPath + '/' + dataType + '/train/' + self.args.dataFile[0]+'_noisy/'+file)
                rec_fbp.tofile(self.args.dataPath + '/' + dataType + '/train/' + self.args.dataFile[2] + '/' + file)
                rec_fbp_noisy.tofile(self.args.dataPath + '/' + dataType + '/train/' + self.args.dataFile[2] + '_noisy/' + file)
                rec_sirt.tofile(self.args.dataPath + '/' + dataType + '/train/' + self.args.dataFile[3] + '/' + file)
                rec_sirt_noisy.tofile( self.args.dataPath + '/' + dataType + '/train/' + self.args.dataFile[3] + '_noisy/' + file)
            print('---------------------------train data generate finish-----------------------------------------')
            print('---------------------------start generate val data-------------------------------------------')
            files = os.listdir(self.args.dataPath+'/'+dataType+'/val/label')
            for step,file in enumerate(files):
                img = np.fromfile(self.args.dataPath+'/'+dataType+'/val/label/'+file,dtype=np.float32).reshape(self.args.nSize,self.args.nSize)
                sinogram, noisy_sinogram, rec_fbp, rec_sirt, rec_fbp_noisy, rec_sirt_noisy = self.projection(img,True)
                sinogram.tofile(self.args.dataPath + '/' + dataType + '/val/' + self.args.dataFile[0]+'/'+file)
                noisy_sinogram.tofile(self.args.dataPath + '/' + dataType + '/val/' + self.args.dataFile[0]+'_noisy/'+file)
                rec_fbp.tofile(self.args.dataPath + '/' + dataType + '/val/' + self.args.dataFile[2] + '/' + file)
                rec_fbp_noisy.tofile(self.args.dataPath + '/' + dataType + '/val/' + self.args.dataFile[2] + '_noisy/' + file)
                rec_sirt.tofile(self.args.dataPath + '/' + dataType + '/val/' + self.args.dataFile[3] + '/' + file)
                rec_sirt_noisy.tofile( self.args.dataPath + '/' + dataType + '/val/' + self.args.dataFile[3] + '_noisy/' + file)
            print('---------------------------validation  data generate finish------------------------------------')

